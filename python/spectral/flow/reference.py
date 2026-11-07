"""Float64 NumPy solver for 2D Navier-Stokes in vorticity form (Task 2.1).

The ground truth for the GPU solver and the backend of the CI-sized data pipeline: plain
NumPy, no GPU, written for clarity over speed.

Pseudo-spectral in space: the state is omega_hat, the rfft2 of the vorticity. Derivatives are
products with i k; the nonlinear term u . grad omega is formed in physical space from four
inverse transforms and brought back with one forward transform, then masked by the 2/3 rule.
The initial condition is masked too, so the state stays on the dealiased modes: the truncated
system then conserves energy and enstrophy exactly in its nonlinear term.

Time stepping, IMEX Crank-Nicolson / Heun, second order: the linear terms L = -nu |k|^2 - alpha
are implicit (Crank-Nicolson), the nonlinear term and the forcing F = f_hat - N_hat explicit
(Heun's predictor-corrector):

    predictor  w* = [(1 + dt/2 L) w + dt F(w)] / (1 - dt/2 L)
    corrector  w' = [(1 + dt/2 L) w + dt/2 (F(w) + F(w*))] / (1 - dt/2 L)

Every function works on a batch: arrays of shape (..., n, n) in physical space, (..., n, n//2+1)
in Fourier space, and viscosity either a float or an array of shape (batch,).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .params import Flow, Grid, forcing, forcing_velocity


def rfft2(x: np.ndarray) -> np.ndarray:
    return np.fft.rfft2(x, axes=(-2, -1))


def irfft2(x: np.ndarray, n: int) -> np.ndarray:
    return np.fft.irfft2(x, s=(n, n), axes=(-2, -1))


def _viscosity(flow: Flow) -> np.ndarray:
    """nu as an array that broadcasts against (..., n, n//2 + 1)."""
    nu = np.asarray(flow.viscosity, dtype=np.float64)
    return nu[..., None, None] if nu.ndim else nu


@dataclass
class Solver:
    """Precomputed operators for one grid and one set of parameters."""

    grid: Grid
    flow: Flow
    dt: float

    def __post_init__(self):
        g = self.grid
        self.ikx = 1j * g.kx
        self.iky = 1j * g.ky
        self.inverse_k2 = g.inverse_k2
        self.mask = g.dealias
        self.forcing_hat = rfft2(forcing(g, self.flow)) * self.mask
        linear = -_viscosity(self.flow) * g.k2 - self.flow.drag  # L, diagonal in Fourier space
        self.explicit = 1.0 + 0.5 * self.dt * linear
        self.implicit = 1.0 / (1.0 - 0.5 * self.dt * linear)

    def velocity_hat(self, w_hat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(u_hat, v_hat) from omega_hat: psi_hat = omega_hat / |k|^2, u = psi_y, v = -psi_x."""
        psi_hat = w_hat * self.inverse_k2
        return self.iky * psi_hat, -self.ikx * psi_hat

    def nonlinear_hat(self, w_hat: np.ndarray) -> np.ndarray:
        """The dealiased Fourier transform of u . grad omega."""
        n = self.grid.n
        u_hat, v_hat = self.velocity_hat(w_hat)
        u = irfft2(u_hat, n)
        v = irfft2(v_hat, n)
        wx = irfft2(self.ikx * w_hat, n)
        wy = irfft2(self.iky * w_hat, n)
        return rfft2(u * wx + v * wy) * self.mask

    def tendency(self, w_hat: np.ndarray) -> np.ndarray:
        """F(omega_hat) = f_hat - N_hat: the explicitly treated part of the right-hand side."""
        return self.forcing_hat - self.nonlinear_hat(w_hat)

    def step(self, w_hat: np.ndarray) -> np.ndarray:
        f0 = self.tendency(w_hat)
        predicted = (self.explicit * w_hat + self.dt * f0) * self.implicit
        f1 = self.tendency(predicted)
        return (self.explicit * w_hat + 0.5 * self.dt * (f0 + f1)) * self.implicit


def to_spectral(w: np.ndarray, grid: Grid) -> np.ndarray:
    """omega_hat of a physical vorticity field, projected onto the dealiased modes."""
    return rfft2(np.asarray(w, dtype=np.float64)) * grid.dealias


def simulate(
    w0: np.ndarray, flow: Flow, dt: float, steps: int, record_every: int = 1
) -> np.ndarray:
    """Vorticity frames from w0 (shape (..., n, n)): frame k is the state after k * record_every
    steps, frame 0 the dealiased initial condition. Returns (frames, ..., n, n), float64."""
    if steps < 0 or record_every < 1:
        raise ValueError("need steps >= 0 and record_every >= 1")
    grid = Grid(np.shape(w0)[-1])
    solver = Solver(grid, flow, dt)
    w_hat = to_spectral(w0, grid)
    frames = [irfft2(w_hat, grid.n)]
    for k in range(1, steps + 1):
        w_hat = solver.step(w_hat)
        if k % record_every == 0:
            frames.append(irfft2(w_hat, grid.n))
    return np.stack(frames)


# --- Diagnostics ------------------------------------------------------------------------------


def diagnostics(w: np.ndarray, flow: Flow) -> dict[str, np.ndarray]:
    """Domain averages of a vorticity field (..., n, n), each of shape (...):

    energy        E = 1/2 <|u|^2>
    enstrophy     Z = 1/2 <omega^2>
    palinstrophy  P = 1/2 <|grad omega|^2>
    energy_input     <u . F>, the work done by the body force
    enstrophy_input  <omega f>

    For the dealiased system the budgets are exact:
        dE/dt = energy_input - 2 nu Z - 2 alpha E
        dZ/dt = enstrophy_input - 2 nu P - 2 alpha Z
    """
    grid = Grid(np.shape(w)[-1])
    solver = Solver(grid, flow, 1.0)
    w_hat = to_spectral(w, grid)
    n = grid.n
    u_hat, v_hat = solver.velocity_hat(w_hat)
    u, v = irfft2(u_hat, n), irfft2(v_hat, n)
    wx, wy = irfft2(solver.ikx * w_hat, n), irfft2(solver.iky * w_hat, n)
    w = irfft2(w_hat, n)

    def mean(a):
        return a.mean(axis=(-2, -1))

    return {
        "energy": 0.5 * mean(u**2 + v**2),
        "enstrophy": 0.5 * mean(w**2),
        "palinstrophy": 0.5 * mean(wx**2 + wy**2),
        "energy_input": mean(u * forcing_velocity(grid, flow)),
        "enstrophy_input": mean(w * forcing(grid, flow)),
    }


def budget_rates(w: np.ndarray, flow: Flow) -> tuple[np.ndarray, np.ndarray]:
    """(dE/dt, dZ/dt) predicted by the budgets in diagnostics()."""
    d = diagnostics(w, flow)
    nu = np.asarray(flow.viscosity, dtype=np.float64)
    de = d["energy_input"] - 2 * nu * d["enstrophy"] - 2 * flow.drag * d["energy"]
    dz = d["enstrophy_input"] - 2 * nu * d["palinstrophy"] - 2 * flow.drag * d["enstrophy"]
    return de, dz
