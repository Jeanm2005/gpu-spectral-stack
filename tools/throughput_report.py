"""Throughput of the data pipeline (Task 2.8): simulations per second, write MB/s, and loader
samples per second, for the dataset params.yaml describes.

    python tools/throughput_report.py --params params.yaml --dataset data/flow.zarr \\
        --out docs/results/throughput.json

    simulation  one batch of batch_size trajectories, spin-up plus every recorded frame, as
                flow.generate runs it, for each FFT backend and spectral layout: trajectories
                per second and solver steps per second (one step advances one trajectory)
    write       finished batches written into a new store and synced to disk: MB/s
    loader      the training loader on the stored dataset after dropping its page cache, at
                several worker counts with no GPU work: windows per second and MB/s

Measure on the charger with the GPU clock locked (docs/DEVELOPMENT.md).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import torch

from spectral.flow.generate import DatasetConfig, generate
from spectral.flow.loader import WindowDataset, make_loader
from spectral.flow.pipeline import load_params
from spectral.flow.store import create

BACKENDS = [("cufft", False), ("cufft", True), ("stockham_r8", False), ("stockham_r8", True)]


def drop_cache(path: Path) -> None:
    os.sync()
    for f in path.rglob("*"):
        if f.is_file():
            fd = os.open(f, os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)


def simulation(config: DatasetConfig) -> list[dict]:
    rows = []
    one_batch = dataclasses.replace(config, num_samples=config.batch_size)
    steps = config.spinup_steps + config.steps
    for fft_impl, pruned in BACKENDS:
        c = dataclasses.replace(one_batch, fft_impl=fft_impl, pruned=pruned)
        warm = dataclasses.replace(c, spinup_steps=0, frames=2, record_every=5)
        next(generate(warm))  # compiles and caches everything the run needs
        torch.cuda.synchronize()
        start = time.perf_counter()
        next(generate(c))
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        rows.append(
            {
                "fft_impl": fft_impl,
                "pruned": pruned,
                "trajectories": c.batch_size,
                "steps_per_trajectory": steps,
                "seconds": round(seconds, 3),
                "trajectories_per_s": round(c.batch_size / seconds, 2),
                "trajectory_steps_per_s": round(c.batch_size * steps / seconds),
            }
        )
        print(rows[-1], file=sys.stderr)
    return rows


def write(config: DatasetConfig, layout: dict, batches: int, workdir: Path | None) -> dict:
    data = next(generate(dataclasses.replace(config, num_samples=config.batch_size)))
    total = dataclasses.replace(config, num_samples=config.batch_size * batches)
    tmp = Path(tempfile.mkdtemp(dir=workdir))
    try:
        ds = create(tmp / "w.zarr", total, **layout)
        os.sync()
        start = time.perf_counter()
        for b in range(batches):
            lo = b * config.batch_size
            ds.vorticity[lo : lo + config.batch_size] = data.vorticity
        os.sync()
        seconds = time.perf_counter() - start
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    nbytes = data.vorticity.nbytes * batches
    return {"mb": round(nbytes / 1e6), "seconds": round(seconds, 3),
            "mb_per_s": round(nbytes / 1e6 / seconds, 1)}  # fmt: skip


def loader(path: Path, batch_size: int, inputs: int, batches: int) -> list[dict]:
    rows = []
    for workers in (0, 2, 4, 8):
        ds = WindowDataset(path, inputs=inputs, outputs=1, seed=0)
        drop_cache(path)
        it = iter(make_loader(ds, batch_size, num_workers=workers))
        start = time.perf_counter()
        for _ in range(batches):
            batch = next(it)
        seconds = time.perf_counter() - start
        frame_bytes = batch["input"].shape[-1] ** 2 * 4
        windows = batches * batch_size
        rows.append(
            {
                "workers": workers,
                "windows_per_s": round(windows / seconds, 1),
                "mb_per_s": round(windows * (inputs + 1) * frame_bytes / 1e6 / seconds, 1),
            }
        )
        print(rows[-1], file=sys.stderr)
        del it
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--params", type=Path, default=Path("params.yaml"))
    parser.add_argument("--dataset", type=Path, default=Path("data/flow.zarr"))
    parser.add_argument("--write-batches", type=int, default=4)
    parser.add_argument("--loader-batches", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--inputs", type=int, default=10)
    parser.add_argument("--workdir", type=Path, default=None)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    params = load_params(args.params)
    config = DatasetConfig.from_dict(params["dataset"])
    report = {
        "device": torch.cuda.get_device_name(),
        "dataset_config": config.as_dict(),
        "simulation": simulation(config),
        "write": write(config, params.get("layout", {}), args.write_batches, args.workdir),
        "loader": loader(args.dataset, args.batch_size, args.inputs, args.loader_batches),
    }
    text = json.dumps(report, indent=1) + "\n"
    if args.out:
        args.out.write_text(text)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
