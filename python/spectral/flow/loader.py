"""Streaming training data from a stored dataset (Task 2.7).

    dataset = WindowDataset("data/flow.zarr", inputs=10, outputs=1, seed=0)
    loader = make_loader(dataset, batch_size=32, num_workers=4)
    for epoch in range(epochs):
        dataset.set_epoch(epoch)
        for batch in CudaPrefetcher(loader):
            batch["input"]   # (32, 10, n, n) float32, on the GPU
            batch["target"]  # (32, 1, n, n)
            batch["viscosity"]

A training item is a window of consecutive frames of one trajectory: `inputs` frames in, the
next `outputs` frames to predict. Every window of every trajectory appears exactly once per
epoch.

How it reads: each DataLoader worker takes a disjoint set of shards (flow.store keeps 16
trajectories per shard) and reads a shard's trajectories in one request, the fastest of the
access patterns in the chunk study (docs/DATA.md). Windows from `shuffle_shards` shards at a time
are mixed in a buffer and yielded in random order, so batches are not dominated by one
trajectory. The order depends only on (seed, epoch), not on timing.

Batches go to pinned (page-locked) host memory, so CudaPrefetcher can copy the next batch to the
GPU asynchronously, on its own stream, while the current one is in use.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from .store import open_dataset


class WindowDataset(IterableDataset):
    def __init__(
        self,
        path: str | Path,
        *,
        inputs: int = 10,
        outputs: int = 1,
        seed: int = 0,
        shuffle: bool = True,
        shuffle_shards: int = 4,
        samples: np.ndarray | None = None,
    ):
        """samples: the trajectories to use (default: all), e.g. a train/test split."""
        self.path = str(path)
        ds = open_dataset(self.path)
        if not ds.complete:
            raise ValueError(f"{path} is incomplete: missing samples {ds.missing()[:10]}...")
        self.frames = ds.config.frames
        self.n = ds.config.n
        if inputs < 1 or outputs < 1 or inputs + outputs > self.frames:
            raise ValueError(f"need inputs, outputs >= 1 and inputs + outputs <= {self.frames}")
        self.inputs, self.outputs = inputs, outputs
        self.seed, self.shuffle, self.shuffle_shards = seed, shuffle, max(1, shuffle_shards)
        self.epoch = 0
        all_samples = np.arange(ds.config.num_samples)
        self.samples = all_samples if samples is None else np.asarray(samples)
        self.per_shard = ds.vorticity.shards[0]
        self.viscosity = ds.viscosity[:]

    @property
    def windows_per_trajectory(self) -> int:
        return self.frames - self.inputs - self.outputs + 1

    def __len__(self) -> int:
        return len(self.samples) * self.windows_per_trajectory

    def set_epoch(self, epoch: int) -> None:
        """Call before each epoch: the order of windows is a function of (seed, epoch)."""
        self.epoch = epoch

    def shards(self) -> list[np.ndarray]:
        """The selected samples grouped by the shard that stores them."""
        ids = self.samples // self.per_shard
        return [self.samples[ids == s] for s in np.unique(ids)]

    def __iter__(self) -> Iterator[dict]:
        rng = np.random.default_rng([self.seed, self.epoch])
        shards = self.shards()
        order = rng.permutation(len(shards)) if self.shuffle else np.arange(len(shards))
        info = get_worker_info()
        if info is not None:  # each worker takes every num_workers-th shard of the order
            order = order[info.id :: info.num_workers]
            rng = np.random.default_rng([self.seed, self.epoch, info.id + 1])
        ds = open_dataset(self.path)  # one handle per worker process
        span = self.inputs + self.outputs
        for group_start in range(0, len(order), self.shuffle_shards):
            items = []
            for s in order[group_start : group_start + self.shuffle_shards]:
                members = shards[s]
                lo, hi = int(members.min()), int(members.max()) + 1
                block = ds.vorticity[lo:hi]  # one read of the shard's trajectories
                for i in members:
                    for start in range(self.windows_per_trajectory):
                        items.append((block, int(i) - lo, int(i), start))
            if self.shuffle:
                items = [items[k] for k in rng.permutation(len(items))]
            for block, row, sample, start in items:
                window = block[row, start : start + span]
                yield {
                    "input": torch.from_numpy(np.ascontiguousarray(window[: self.inputs])),
                    "target": torch.from_numpy(np.ascontiguousarray(window[self.inputs :])),
                    "viscosity": torch.tensor(float(self.viscosity[sample])),
                    "sample": torch.tensor(sample),
                    "start": torch.tensor(start),
                }


def make_loader(
    dataset: WindowDataset,
    batch_size: int,
    *,
    num_workers: int = 4,
    pin_memory: bool = True,
    prefetch_factor: int = 4,
) -> DataLoader:
    """A DataLoader over `dataset`: worker processes read and assemble batches ahead of time
    (prefetch_factor batches each), into pinned memory when pin_memory is set."""
    extra = {}
    if num_workers > 0:
        extra = {"prefetch_factor": prefetch_factor, "persistent_workers": True}
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        **extra,
    )


class CudaPrefetcher:
    """Wraps a loader: while the caller works on batch k, batch k + 1 is already being copied
    to the GPU on a separate stream. Copies are asynchronous only from pinned memory."""

    def __init__(self, loader, device: torch.device | str = "cuda"):
        self.loader = loader
        self.device = torch.device(device)
        self.stream = torch.cuda.Stream(device=self.device)

    def _to_device(self, batch: dict) -> dict:
        with torch.cuda.stream(self.stream):
            return {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

    def __len__(self) -> int:
        return len(self.loader)

    def __iter__(self) -> Iterator[dict]:
        it = iter(self.loader)
        try:
            pending = self._to_device(next(it))
        except StopIteration:
            return
        current_stream = torch.cuda.current_stream(self.device)
        for nxt in it:
            current_stream.wait_stream(self.stream)  # the pending copy must finish first
            ready = pending
            for v in ready.values():
                v.record_stream(current_stream)  # memory stays valid while used here
            pending = self._to_device(nxt)
            yield ready
        current_stream.wait_stream(self.stream)
        for v in pending.values():
            v.record_stream(current_stream)
        yield pending


def batches_per_epoch(dataset: WindowDataset, batch_size: int) -> int:
    return math.ceil(len(dataset) / batch_size)
