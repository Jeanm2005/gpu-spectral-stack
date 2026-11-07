"""Batched generation of flow trajectories (Task 2.3).

A dataset is fully described by a DatasetConfig. Its hash (config_hash) names the dataset, and
every sample i gets its own seed, derived from (config.seed, i), from which its viscosity and
its initial condition are drawn. Neither depends on the batch size or on which batch a sample
falls in, so any sample can be regenerated on its own, and a dataset regenerates identically
whatever batch size or GPU it runs on.

Initial conditions are Gaussian random fields: white noise filtered to the spectrum
(|k|^2 + tau^2)^(-alpha/2), the prior used for the FNO Navier-Stokes benchmarks, with zero mean
and a fixed RMS. They are drawn in float64 NumPy on the CPU (cheap next to the simulation), so
the GPU and NumPy backends start from identical fields.

Each batch of trajectories advances together: on the GPU in one GpuSolver call per step, with a
viscosity per trajectory; with backend "numpy", through the float64 reference solver (slow; for
CI-sized datasets).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field, fields

import numpy as np

from . import reference
from .params import Flow, Grid

SCHEMA_VERSION = 1  # bump when the meaning of a config field or of the output changes


@dataclass(frozen=True)
class DatasetConfig:
    """Everything that determines a dataset's contents."""

    n: int = 64  # grid points per side
    dt: float = 5e-3  # time step
    record_every: int = 20  # steps between recorded frames
    frames: int = 21  # recorded frames per trajectory, the initial condition included
    spinup_steps: int = 0  # steps run before the first recorded frame
    num_samples: int = 16  # trajectories
    seed: int = 0

    viscosity_min: float = 1e-3  # viscosity is log-uniform in [min, max]
    viscosity_max: float = 1e-3
    drag: float = 0.1
    forcing_amplitude: float = 1.0
    forcing_wavenumber: int = 4

    ic_alpha: float = 2.5  # Gaussian random field: spectrum (|k|^2 + tau^2)^(-alpha / 2)
    ic_tau: float = 7.0
    ic_rms: float = 5.0  # RMS vorticity of the initial condition

    backend: str = "gpu"  # "gpu" or "numpy"
    fft_impl: str = "stockham_r8"  # GPU backend only: "cufft" or one of spectral.FFT_IMPLS
    pruned: bool = True  # GPU backend only: transforms on the dealiased modes only

    # Not part of the dataset's identity: how many trajectories run together.
    batch_size: int = field(default=16, metadata={"identity": False})

    def __post_init__(self):
        Grid(self.n)  # validates n
        checks = {
            "dt > 0": self.dt > 0,
            "record_every >= 1": self.record_every >= 1,
            "frames >= 1": self.frames >= 1,
            "spinup_steps >= 0": self.spinup_steps >= 0,
            "num_samples >= 1": self.num_samples >= 1,
            "batch_size >= 1": self.batch_size >= 1,
            "0 < viscosity_min <= viscosity_max": 0 < self.viscosity_min <= self.viscosity_max,
            "backend is gpu or numpy": self.backend in ("gpu", "numpy"),
            "ic_rms > 0": self.ic_rms > 0,
        }
        failed = [name for name, ok in checks.items() if not ok]
        if failed:
            raise ValueError(f"invalid dataset config: need {', '.join(failed)}")

    @property
    def steps(self) -> int:
        """Steps after spin-up: frames - 1 intervals of record_every steps."""
        return (self.frames - 1) * self.record_every

    @property
    def times(self) -> np.ndarray:
        """Simulation time of each recorded frame."""
        return (self.spinup_steps + self.record_every * np.arange(self.frames)) * self.dt

    def identity(self) -> dict:
        """The fields that determine the data, plus the schema version."""
        out = {
            f.name: getattr(self, f.name) for f in fields(self) if f.metadata.get("identity", True)
        }
        out["schema_version"] = SCHEMA_VERSION
        return out

    @classmethod
    def from_dict(cls, d: dict) -> DatasetConfig:
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known - {"schema_version"}
        if unknown:
            raise ValueError(f"unknown dataset config keys: {sorted(unknown)}")
        return cls(**{k: v for k, v in d.items() if k in known})

    def as_dict(self) -> dict:
        return asdict(self)


def config_hash(config: DatasetConfig) -> str:
    """SHA-256 of the config's identity as canonical JSON (sorted keys, no whitespace)."""
    text = json.dumps(config.identity(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def sample_seed(config: DatasetConfig, index: int) -> int:
    """The 32-bit seed of sample `index`, independent of every other sample."""
    return int(np.random.SeedSequence([config.seed, index]).generate_state(1)[0])


def sample_viscosity(config: DatasetConfig, rng: np.random.Generator) -> float:
    lo, hi = np.log(config.viscosity_min), np.log(config.viscosity_max)
    return float(np.exp(rng.uniform(lo, hi))) if hi > lo else config.viscosity_min


def gaussian_random_field(config: DatasetConfig, rng: np.random.Generator) -> np.ndarray:
    """An (n, n) float64 initial vorticity: white noise filtered to (|k|^2 + tau^2)^(-alpha/2),
    zero mean, RMS config.ic_rms."""
    grid = Grid(config.n)
    noise_hat = reference.rfft2(rng.standard_normal((config.n, config.n)))
    noise_hat *= (grid.k2 + config.ic_tau**2) ** (-config.ic_alpha / 2.0)
    noise_hat[0, 0] = 0.0
    w = reference.irfft2(noise_hat, config.n)
    return w * (config.ic_rms / w.std())


@dataclass
class Batch:
    """One batch of finished trajectories."""

    indices: np.ndarray  # (b,) sample indices
    seeds: np.ndarray  # (b,) uint32
    viscosity: np.ndarray  # (b,) float64
    vorticity: np.ndarray  # (b, frames, n, n) float32


def draw_samples(config: DatasetConfig, indices: np.ndarray):
    """(seeds, viscosities, initial conditions) of the given samples."""
    seeds = np.array([sample_seed(config, int(i)) for i in indices], dtype=np.uint32)
    viscosity = np.empty(len(indices))
    w0 = np.empty((len(indices), config.n, config.n))
    for b, seed in enumerate(seeds):
        rng = np.random.default_rng(int(seed))
        viscosity[b] = sample_viscosity(config, rng)
        w0[b] = gaussian_random_field(config, rng)
    return seeds, viscosity, w0


def _simulate_gpu(config: DatasetConfig, flow: Flow, w0: np.ndarray) -> np.ndarray:
    import torch

    from .gpu import GpuSolver

    solver = GpuSolver(config.n, flow, config.dt, fft_impl=config.fft_impl, pruned=config.pruned)
    w_hat = solver.to_spectral(torch.as_tensor(w0, dtype=torch.float32, device="cuda"))
    for _ in range(config.spinup_steps):
        w_hat = solver.step(w_hat)
    w = solver.to_physical(w_hat)
    out = torch.empty((config.frames, *w.shape), dtype=torch.float32, device="cuda")
    out[0] = w
    for k in range(1, config.frames):
        for _ in range(config.record_every):
            w_hat = solver.step(w_hat)
        out[k] = solver.to_physical(w_hat)
    return out.transpose(0, 1).cpu().numpy()  # (b, frames, n, n)


def _simulate_numpy(config: DatasetConfig, flow: Flow, w0: np.ndarray) -> np.ndarray:
    if config.spinup_steps:
        w0 = reference.simulate(
            w0, flow, config.dt, config.spinup_steps, record_every=config.spinup_steps
        )[-1]
    frames = reference.simulate(w0, flow, config.dt, config.steps, config.record_every)
    return np.moveaxis(frames, 0, 1).astype(np.float32)


def generate(config: DatasetConfig, start: int = 0, stop: int | None = None) -> Iterator[Batch]:
    """Simulates samples start..stop-1 (default: all), batch_size at a time, in order."""
    stop = config.num_samples if stop is None else stop
    for first in range(start, stop, config.batch_size):
        indices = np.arange(first, min(first + config.batch_size, stop))
        seeds, viscosity, w0 = draw_samples(config, indices)
        flow = Flow(
            viscosity=viscosity,
            drag=config.drag,
            forcing_amplitude=config.forcing_amplitude,
            forcing_wavenumber=config.forcing_wavenumber,
        )
        simulate = _simulate_gpu if config.backend == "gpu" else _simulate_numpy
        yield Batch(indices, seeds, viscosity, simulate(config, flow, w0))
