"""The streaming loader (Task 2.7): window contents, every window exactly once per epoch with or
without workers, reproducible order, and the CUDA prefetcher."""

import numpy as np
import pytest
import torch

from spectral.flow.generate import DatasetConfig
from spectral.flow.loader import CudaPrefetcher, WindowDataset, make_loader
from spectral.flow.store import create, write_dataset

CONFIG = DatasetConfig(
    n=16, dt=0.01, record_every=2, frames=8, num_samples=10, batch_size=10, backend="numpy"
)


@pytest.fixture(scope="module")
def path(tmp_path_factory):
    p = tmp_path_factory.mktemp("loader") / "d.zarr"
    write_dataset(p, CONFIG, samples_per_shard=4)  # shards of 4, 4, 2 trajectories
    return p


def keys(items):
    return sorted((int(s), int(t)) for s, t in items)


def all_windows(dataset):
    return keys((s, t) for s in dataset.samples for t in range(dataset.windows_per_trajectory))


def test_windows_hold_the_right_frames(path):
    from spectral.flow.store import open_dataset

    stored = open_dataset(path)
    ds = WindowDataset(path, inputs=3, outputs=2)
    for item in list(ds)[:20]:
        i, t = int(item["sample"]), int(item["start"])
        np.testing.assert_array_equal(item["input"].numpy(), stored.vorticity[i, t : t + 3])
        np.testing.assert_array_equal(item["target"].numpy(), stored.vorticity[i, t + 3 : t + 5])
        assert float(item["viscosity"]) == pytest.approx(stored.viscosity[i])
        assert item["input"].dtype == torch.float32


def test_every_window_once_per_epoch(path):
    ds = WindowDataset(path, inputs=3, outputs=1)
    seen = keys((it["sample"], it["start"]) for it in ds)
    assert seen == all_windows(ds)
    assert len(seen) == len(ds) == 10 * 5


@pytest.mark.parametrize("workers", [0, 2, 3])
def test_every_window_once_per_epoch_with_workers(path, workers):
    ds = WindowDataset(path, inputs=3, outputs=1)
    loader = make_loader(ds, batch_size=7, num_workers=workers, pin_memory=False)
    seen = [(s, t) for b in loader for s, t in zip(b["sample"], b["start"], strict=True)]
    assert keys(seen) == all_windows(ds)


def test_order_is_reproducible_and_changes_with_the_epoch(path):
    def order(epoch, seed=0):
        ds = WindowDataset(path, seed=seed, inputs=2)
        ds.set_epoch(epoch)
        return [(int(it["sample"]), int(it["start"])) for it in ds]

    assert order(0) == order(0)
    assert order(0) != order(1)
    assert order(0) != order(0, seed=1)
    assert sorted(order(0)) == sorted(order(1))


def test_windows_mix_trajectories(path):
    ds = WindowDataset(path, inputs=2, shuffle_shards=3)
    first = [int(it["sample"]) for it in list(ds)[:16]]
    assert len(set(first)) > 3  # not one trajectory's windows in a row


def test_unshuffled_order_is_sequential(path):
    ds = WindowDataset(path, inputs=2, shuffle=False)
    got = [(int(it["sample"]), int(it["start"])) for it in ds]
    assert got == sorted(got)


def test_a_subset_of_samples(path):
    ds = WindowDataset(path, inputs=2, samples=np.array([1, 5, 9]))
    assert {int(it["sample"]) for it in ds} == {1, 5, 9}
    assert len(ds) == 3 * ds.windows_per_trajectory


def test_rejects_windows_longer_than_a_trajectory(path):
    with pytest.raises(ValueError, match="inputs \\+ outputs"):
        WindowDataset(path, inputs=8, outputs=1)


def test_rejects_an_incomplete_dataset(tmp_path):
    create(tmp_path / "d.zarr", CONFIG)
    with pytest.raises(ValueError, match="incomplete"):
        WindowDataset(tmp_path / "d.zarr")


def test_batches_come_out_pinned(path):
    if not torch.cuda.is_available():
        pytest.skip("pinning needs CUDA")
    batch = next(iter(make_loader(WindowDataset(path, inputs=3), batch_size=4, num_workers=0)))
    assert batch["input"].is_pinned()


@pytest.mark.gpu
@pytest.mark.parametrize("workers", [0, 2])
def test_cuda_prefetcher_delivers_the_same_batches_on_the_gpu(path, workers):
    ds = WindowDataset(path, inputs=3)
    plain = list(make_loader(ds, batch_size=8, num_workers=workers))
    fetched = list(CudaPrefetcher(make_loader(ds, batch_size=8, num_workers=workers)))
    assert len(fetched) == len(plain)
    for a, b in zip(plain, fetched, strict=True):
        assert b["input"].is_cuda
        torch.testing.assert_close(b["input"].cpu(), a["input"], rtol=0, atol=0)
        torch.testing.assert_close(b["sample"].cpu(), a["sample"], rtol=0, atol=0)
