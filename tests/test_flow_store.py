"""Datasets on disk (Task 2.4): the Zarr layout, the metadata schema, and resuming. CPU only (the
NumPy backend), so CI runs it."""

import dataclasses
import subprocess

import numpy as np
import pytest
import zarr

from spectral.flow.generate import DatasetConfig, config_hash, generate
from spectral.flow.store import (
    METADATA_SCHEMA,
    create,
    open_dataset,
    schema_errors,
    write_dataset,
)

CONFIG = DatasetConfig(
    n=16, dt=0.01, record_every=4, frames=5, num_samples=6, batch_size=4, backend="numpy"
)


def expected_vorticity(config):
    return np.concatenate([b.vorticity for b in generate(config)])


def test_round_trip(tmp_path):
    ds = write_dataset(tmp_path / "d.zarr", CONFIG)
    assert ds.complete
    back = open_dataset(tmp_path / "d.zarr")
    np.testing.assert_array_equal(back.vorticity[:], expected_vorticity(CONFIG))
    np.testing.assert_allclose(back.time[:], CONFIG.times)
    assert back.seed[:].dtype == np.uint32 and np.all(back.seed[:] > 0)
    assert back.config == CONFIG
    assert back.attrs["config_hash"] == config_hash(CONFIG)


def test_layout_shards_and_chunks(tmp_path):
    create(tmp_path / "d.zarr", CONFIG, chunk_frames=2, samples_per_shard=4)
    arr = zarr.open_array(str(tmp_path / "d.zarr" / "vorticity"), mode="r")
    assert arr.shape == (6, 5, 16, 16)
    assert arr.chunks == (1, 2, 16, 16)
    assert arr.shards == (4, 6, 16, 16)  # frames rounded up to whole chunks
    assert arr.metadata.dimension_names == ("sample", "frame", "y", "x")


def test_metadata_follows_the_schema(tmp_path):
    ds = create(tmp_path / "d.zarr", CONFIG)
    assert schema_errors(ds.attrs) == []
    assert set(METADATA_SCHEMA) <= set(ds.attrs)
    assert ds.attrs["resolution"] == 16 and ds.attrs["solver"]["backend"] == "numpy"


def test_records_the_git_commit(tmp_path):
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    ds = create(tmp_path / "d.zarr", CONFIG)
    assert ds.attrs["git_commit"].startswith(head)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda a: a.pop("dt"), "missing dt"),
        (lambda a: a.update(resolution="16"), "resolution should be int"),
        (lambda a: a.update(samples=True), "samples should be int"),
        (lambda a: a.update(config_hash="0" * 64), "config_hash does not match"),
        (lambda a: a.update(frames=99), "frames 99 disagrees"),
        (lambda a: a.update(schema_version=99), "schema_version 99"),
    ],
)
def test_schema_violations_are_reported(tmp_path, change, message):
    attrs = dict(create(tmp_path / "d.zarr", CONFIG).attrs)
    change(attrs)
    assert any(message in e for e in schema_errors(attrs))


def test_opening_a_dataset_with_bad_metadata_fails(tmp_path):
    ds = create(tmp_path / "d.zarr", CONFIG)
    ds.group.attrs["config_hash"] = "0" * 64
    with pytest.raises(ValueError, match="invalid dataset metadata"):
        open_dataset(tmp_path / "d.zarr")


def test_an_interrupted_generation_resumes(tmp_path):
    path = tmp_path / "d.zarr"
    ds = create(path, CONFIG)
    first = next(generate(CONFIG))  # samples 0-3 written, then the run "dies"
    ds.write(first)
    assert not ds.complete and open_dataset(path).missing().tolist() == [4, 5]

    resumed = write_dataset(path, CONFIG)
    assert resumed.complete
    np.testing.assert_array_equal(resumed.vorticity[:], expected_vorticity(CONFIG))


def test_refuses_to_resume_a_different_dataset(tmp_path):
    write_dataset(tmp_path / "d.zarr", CONFIG)
    with pytest.raises(ValueError, match="different dataset"):
        write_dataset(tmp_path / "d.zarr", dataclasses.replace(CONFIG, seed=1))


def test_create_refuses_an_existing_path(tmp_path):
    create(tmp_path / "d.zarr", CONFIG)
    with pytest.raises(FileExistsError):
        create(tmp_path / "d.zarr", CONFIG)


def test_unwritten_samples_read_as_nan(tmp_path):
    ds = create(tmp_path / "d.zarr", CONFIG)
    assert np.isnan(ds.vorticity[0]).all()


def test_compression_round_trips(tmp_path):
    ds = write_dataset(tmp_path / "d.zarr", CONFIG, compression=True)
    np.testing.assert_array_equal(ds.vorticity[:], expected_vorticity(CONFIG))
