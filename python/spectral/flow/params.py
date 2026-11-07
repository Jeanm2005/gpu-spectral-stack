"""Physical parameters and the spectral grid, shared by every solver.

Fields are real arrays of shape (..., n, n) holding values at x_j = 2 pi j / n (last axis) and
y_i = 2 pi i / n (second-to-last axis). Their spectra are in the rfft2 layout, (..., n, n//2 + 1):
rows are ky = 0, 1, ..., n/2 - 1, -n/2, ..., -1 and columns kx = 0, 1, ..., n/2.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class Flow:
    """The equation's parameters. viscosity may differ per trajectory: solvers accept an array
    of shape (batch,) in its place."""

    viscosity: float = 1e-3  # nu
    drag: float = 0.1  # alpha, linear friction: removes energy at large scales
    forcing_amplitude: float = 1.0  # A
    forcing_wavenumber: int = 4  # k_f

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Grid:
    """Wavenumbers of an n x n grid in the rfft2 layout, and the 2/3-rule dealiasing mask."""

    n: int

    def __post_init__(self):
        if self.n < 4 or self.n & (self.n - 1):
            raise ValueError(f"n must be a power of two of at least 4, got {self.n}")

    @property
    def kx(self) -> np.ndarray:
        """(1, n//2 + 1): wavenumbers along x, the last axis."""
        return np.fft.rfftfreq(self.n, 1.0 / self.n)[None, :]

    @property
    def ky(self) -> np.ndarray:
        """(n, 1): wavenumbers along y."""
        return np.fft.fftfreq(self.n, 1.0 / self.n)[:, None]

    @property
    def k2(self) -> np.ndarray:
        """(n, n//2 + 1): |k|^2."""
        return self.kx**2 + self.ky**2

    @property
    def inverse_k2(self) -> np.ndarray:
        """1 / |k|^2, with 0 for the mean mode (the streamfunction has no mean)."""
        k2 = self.k2
        out = np.zeros_like(k2)
        out[k2 > 0] = 1.0 / k2[k2 > 0]
        return out

    @property
    def dealias(self) -> np.ndarray:
        """(n, n//2 + 1) of bool: modes kept by the 2/3 rule, |kx| and |ky| at most n/3. A
        quadratic product of fields inside this set aliases only onto modes outside it, so
        masking the nonlinear term computes it exactly on the kept modes."""
        kmax = self.n / 3.0
        return (np.abs(self.kx) <= kmax) & (np.abs(self.ky) <= kmax)

    @property
    def y(self) -> np.ndarray:
        """(n, 1): grid points along y."""
        return (2.0 * np.pi * np.arange(self.n) / self.n)[:, None]

    @property
    def x(self) -> np.ndarray:
        """(1, n): grid points along x."""
        return (2.0 * np.pi * np.arange(self.n) / self.n)[None, :]


def forcing(grid: Grid, flow: Flow) -> np.ndarray:
    """f(y) = -A k_f cos(k_f y) on the grid, shape (n, n)."""
    k = flow.forcing_wavenumber
    f = -flow.forcing_amplitude * k * np.cos(k * grid.y)
    return np.broadcast_to(f, (grid.n, grid.n)).copy()


def forcing_velocity(grid: Grid, flow: Flow) -> np.ndarray:
    """The x-component of the body force, A sin(k_f y), shape (n, n); its curl is forcing()."""
    f = flow.forcing_amplitude * np.sin(flow.forcing_wavenumber * grid.y)
    return np.broadcast_to(f, (grid.n, grid.n)).copy()


def laminar_kolmogorov(grid: Grid, flow: Flow) -> np.ndarray:
    """The steady laminar solution: a pure shear flow, for which the nonlinear term vanishes,
    so nu k_f^2 omega + alpha omega = f, omega = f / (nu k_f^2 + alpha)."""
    viscosity = float(np.asarray(flow.viscosity))
    return forcing(grid, flow) / (viscosity * flow.forcing_wavenumber**2 + flow.drag)


def taylor_green(grid: Grid, t: float, viscosity: float, drag: float) -> np.ndarray:
    """The unforced Taylor-Green vortex omega = 2 sin x sin y exp(-(2 nu + alpha) t): psi is
    proportional to omega, so u . grad omega = 0 and the decay is exact."""
    return 2.0 * np.sin(grid.x) * np.sin(grid.y) * np.exp(-(2.0 * viscosity + drag) * t)
