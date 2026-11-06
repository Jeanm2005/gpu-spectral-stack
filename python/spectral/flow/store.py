"""Datasets on disk: sharded Zarr v3 with a checked metadata schema (Task 2.4).

A dataset is a Zarr group (a directory):

    vorticity  (samples, frames, n, n) float32   the trajectories
    viscosity  (samples,) float64                 each trajectory's nu
    seed       (samples,) uint32                  each trajectory's seed (flow.generate)
    written    (samples,) bool                    which samples are on disk
    time       (frames,) float64                  simulation time of each frame

vorticity is stored in shards of `samples_per_shard` trajectories (one file each), and inside a
shard in chunks of (1, chunk_frames, n, n): a reader fetches only the chunks it needs from a
shard, while the file count stays at samples / samples_per_shard. docs/DATA.md has the chunk
study behind the defaults.

The group's attributes follow METADATA_SCHEMA and are checked whenever a dataset is opened.
`written` lets an interrupted generation resume and lets validation (Task 2.5) refuse a dataset
with missing samples.
"""

from __future__ import annotations

import datetime
import subprocess
from pathlib import Path

import numpy as np
import zarr
from zarr.codecs import ZstdCodec

from .generate import SCHEMA_VERSION, Batch, DatasetConfig, config_hash

# key -> (type, description). Every key is required.
METADATA_SCHEMA: dict[str, tuple[type | tuple[type, ...], str]] = {
    "schema_version": (int, "version of this layout and of the config fields"),
    "config": (dict, "the DatasetConfig that produced the data"),
    "config_hash": (str, "SHA-256 of the config's identity (flow.generate.config_hash)"),
    "git_commit": (str, "commit of the generating code; '-dirty' if it had local changes"),
    "created_utc": (str, "ISO 8601 time the dataset was created"),
    "generator": (str, "module that wrote the data"),
    "equation": (str, "the PDE, in words"),
    "domain": (str, "the spatial domain and its boundary conditions"),
    "resolution": (int, "grid points per side, n"),
    "dt": ((int, float), "solver time step"),
    "record_every": (int, "solver steps between recorded frames"),
    "viscosity_range": (list, "[min, max] of the log-uniform viscosity distribution"),
    "samples": (int, "number of trajectories"),
    "frames": (int, "frames per trajectory"),
    "solver": (dict, "backend, fft_impl, pruned"),
}

EQUATION = (
    "d omega/dt + u . grad omega = nu lap omega - alpha omega - A k_f cos(k_f y), "
    "u = (psi_y, -psi_x), lap psi = -omega"
)
DOMAIN = "[0, 2 pi)^2, periodic; x along the last axis, y along the second to last"


def git_commit() -> str:
    """HEAD of the repository this code runs from, with '-dirty' for uncommitted changes."""
    root = Path(__file__).resolve().parent
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return head + ("-dirty" if dirty else "")


def metadata(config: DatasetConfig) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "config": config.as_dict(),
        "config_hash": config_hash(config),
        "git_commit": git_commit(),
        "created_utc": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "generator": "spectral.flow.generate",
        "equation": EQUATION,
        "domain": DOMAIN,
        "resolution": config.n,
        "dt": config.dt,
        "record_every": config.record_every,
        "viscosity_range": [config.viscosity_min, config.viscosity_max],
        "samples": config.num_samples,
        "frames": config.frames,
        "solver": {
            "backend": config.backend,
            "fft_impl": config.fft_impl,
            "pruned": config.pruned,
        },
    }


def schema_errors(attrs: dict) -> list[str]:
    """Every way `attrs` departs from METADATA_SCHEMA and from itself (empty when valid)."""
    errors = []
    for key, (kind, _) in METADATA_SCHEMA.items():
        if key not in attrs:
            errors.append(f"missing {key}")
            continue
        value = attrs[key]
        if isinstance(value, bool) or not isinstance(value, kind):  # bool is an int in Python
            errors.append(f"{key} should be {getattr(kind, '__name__', kind)}")
    if errors:
        return errors
    if attrs["schema_version"] != SCHEMA_VERSION:
        errors.append(f"schema_version {attrs['schema_version']}, this code reads {SCHEMA_VERSION}")
        return errors
    try:
        config = DatasetConfig.from_dict(attrs["config"])
    except (TypeError, ValueError) as e:
        return [f"config: {e}"]
    if config_hash(config) != attrs["config_hash"]:
        errors.append("config_hash does not match config")
    for key, value in [
        ("resolution", config.n),
        ("frames", config.frames),
        ("samples", config.num_samples),
        ("dt", config.dt),
        ("record_every", config.record_every),
    ]:
        if attrs[key] != value:
            errors.append(f"{key} {attrs[key]} disagrees with config ({value})")
    return errors


class Dataset:
    """An open dataset. vorticity, viscosity, seed, time, written are its Zarr arrays."""

    def __init__(self, group: zarr.Group):
        errors = schema_errors(dict(group.attrs))
        if errors:
            raise ValueError("invalid dataset metadata: " + "; ".join(errors))
        self.group = group
        self.attrs = dict(group.attrs)
        self.config = DatasetConfig.from_dict(self.attrs["config"])
        self.vorticity = group["vorticity"]
        self.viscosity = group["viscosity"]
        self.seed = group["seed"]
        self.time = group["time"]
        self.written = group["written"]

    @property
    def complete(self) -> bool:
        return bool(np.all(self.written[:]))

    def missing(self) -> np.ndarray:
        return np.flatnonzero(~self.written[:])

    def write(self, batch: Batch) -> None:
        """Stores a batch of finished trajectories (contiguous sample indices)."""
        lo, hi = int(batch.indices[0]), int(batch.indices[-1]) + 1
        if not np.array_equal(batch.indices, np.arange(lo, hi)):
            raise ValueError("a batch must hold contiguous sample indices")
        self.vorticity[lo:hi] = batch.vorticity
        self.viscosity[lo:hi] = batch.viscosity
        self.seed[lo:hi] = batch.seeds
        self.written[lo:hi] = True


def create(
    path: str | Path,
    config: DatasetConfig,
    *,
    chunk_frames: int = 8,
    samples_per_shard: int = 16,
    compression: bool = False,
) -> Dataset:
    """A new, empty dataset for `config` at `path` (which must not exist).

    The defaults come from the chunk study in docs/DATA.md: chunks of 8 frames read fastest for
    every training access pattern; zstd saves 8 % of the space and halves read speed, so it is
    off. A frame count that is a multiple of chunk_frames avoids a padded last chunk."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} already exists")
    s, t, n = config.num_samples, config.frames, config.n
    chunk_frames = min(chunk_frames, t)
    per_shard = min(samples_per_shard, s)
    group = zarr.open_group(str(path), mode="w-", zarr_format=3)
    group.create_array(
        "vorticity",
        shape=(s, t, n, n),
        chunks=(1, chunk_frames, n, n),
        shards=(per_shard, chunk_frames * -(-t // chunk_frames), n, n),
        dtype="float32",
        compressors=[ZstdCodec(level=1)] if compression else None,
        fill_value=np.nan,
        dimension_names=["sample", "frame", "y", "x"],
    )
    group.create_array("viscosity", shape=(s,), dtype="float64", fill_value=np.nan)
    group.create_array("seed", shape=(s,), dtype="uint32", fill_value=0)
    group.create_array("written", shape=(s,), dtype="bool", fill_value=False)
    time = group.create_array("time", shape=(t,), dtype="float64")
    time[:] = config.times
    group.attrs.update(metadata(config))
    return Dataset(group)


def open_dataset(path: str | Path, mode: str = "r") -> Dataset:
    """Opens and checks a dataset; mode "r+" to keep writing (resuming a generation)."""
    return Dataset(zarr.open_group(str(path), mode=mode, zarr_format=3))


def write_dataset(path: str | Path, config: DatasetConfig, **layout) -> Dataset:
    """Generates `config` into a dataset at `path`. If `path` already holds a dataset of the
    same config (same hash), only its missing samples are generated: an interrupted run
    resumes. `layout` goes to create()."""
    from .generate import generate

    path = Path(path)
    if path.exists():
        dataset = open_dataset(path, mode="r+")
        if dataset.attrs["config_hash"] != config_hash(config):
            raise ValueError(f"{path} holds a different dataset (config hash differs)")
    else:
        dataset = create(path, config, **layout)
    missing = dataset.missing()
    # Contiguous runs of missing samples, each generated batch by batch.
    runs = np.split(missing, np.flatnonzero(np.diff(missing) != 1) + 1) if missing.size else []
    for run in runs:
        for batch in generate(config, start=int(run[0]), stop=int(run[-1]) + 1):
            dataset.write(batch)
    return dataset
