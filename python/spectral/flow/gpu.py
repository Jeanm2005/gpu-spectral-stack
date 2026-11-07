"""GPU pseudo-spectral solver for 2D Navier-Stokes in vorticity form, on our FFTs (Task 2.2).

The same discretization as flow.reference (read its docstring first): rfft2 layout, 2/3-rule
dealiasing, IMEX Crank-Nicolson / Heun. Differences:

- float32 on the GPU, a batch of trajectories advanced together, one viscosity each if wanted
- the FFT backend is selectable: fft_impl="cufft" (torch.fft) or one of our kernels
- pruned=True keeps the state only on the modes the 2/3 rule keeps, stored in the layout of
  spectral.rfft2_truncated: rows ky = 0..m-1 and -m..-1, columns kx = 0..m-1 with m = n/3 + 1.
  The transforms then move only those modes (Task 1.7). The row ky = -m is outside the 2/3 set
  and stays masked, so the result is the same as the full layout's.

Each step evaluates the nonlinear term twice; each evaluation is one batched inverse transform
of four fields (u, v, d omega/dx, d omega/dy) and one forward transform of their combination.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import torch

from .. import ops
from .params import Flow, Grid, forcing


class Spectral:
    """A forward / inverse real 2D transform pair and the wavenumbers of its layout."""

    def __init__(self, n: int, fft_impl: str, pruned: bool, device: torch.device | str):
        if fft_impl != "cufft" and fft_impl not in ops.FFT_IMPLS:
            raise ValueError(
                f"unknown fft_impl '{fft_impl}' (available: cufft, {', '.join(ops.FFT_IMPLS)})"
            )
        self.n, self.fft_impl, self.pruned = n, fft_impl, pruned
        grid = Grid(n)
        kmax = n // 3
        ky_full = torch.as_tensor(grid.ky[:, 0])
        kx_full = torch.as_tensor(grid.kx[0])
        if pruned:
            self.m = kmax + 1
            ky = torch.cat([ky_full[: self.m], ky_full[n - self.m :]])
            kx = kx_full[: self.m]
        else:
            ky, kx = ky_full, kx_full
        self.ky = ky[:, None].to(device=device, dtype=torch.float32)
        self.kx = kx[None, :].to(device=device, dtype=torch.float32)
        self.mask = ((self.kx.abs() <= n / 3) & (self.ky.abs() <= n / 3)).to(torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Real (..., n, n) float32 -> complex64 spectrum in this layout."""
        if self.fft_impl == "cufft":
            full = torch.fft.rfft2(x)
            if not self.pruned:
                return full
            m, n = self.m, self.n
            return torch.cat([full[..., :m, :m], full[..., n - m :, :m]], dim=-2)
        if self.pruned:
            return ops.rfft2_truncated(x, self.m, self.m, impl=self.fft_impl)
        return ops.rfft2(x, impl=self.fft_impl)

    def inverse(self, x_hat: torch.Tensor) -> torch.Tensor:
        """Complex64 spectrum in this layout -> real (..., n, n) float32."""
        n = self.n
        if self.fft_impl == "cufft":
            if self.pruned:
                m = self.m
                full = x_hat.new_zeros((*x_hat.shape[:-2], n, n // 2 + 1))
                full[..., :m, :m] = x_hat[..., :m, :]
                full[..., n - m :, :m] = x_hat[..., m:, :]
                x_hat = full
            return torch.fft.irfft2(x_hat, s=(n, n))
        if self.pruned:
            return ops.irfft2_from_modes(x_hat, (n, n), impl=self.fft_impl)
        return ops.irfft2(x_hat, n, impl=self.fft_impl)


class GpuSolver:
    """Advances a batch of vorticity fields. viscosity: a float, or one value per trajectory."""

    def __init__(
        self,
        n: int,
        flow: Flow,
        dt: float,
        *,
        fft_impl: str = "stockham_r8",
        pruned: bool = False,
        device: torch.device | str = "cuda",
    ):
        self.n, self.flow, self.dt = n, flow, dt
        self.spectral = s = Spectral(n, fft_impl, pruned, device)
        self.ikx = (1j * s.kx).to(torch.complex64)
        self.iky = (1j * s.ky).to(torch.complex64)
        k2 = s.kx**2 + s.ky**2
        self.inverse_k2 = torch.where(k2 > 0, 1.0 / k2.clamp(min=1.0), torch.zeros_like(k2))

        f = torch.as_tensor(forcing(Grid(n), flow), dtype=torch.float32, device=device)
        self.forcing_hat = s.forward(f) * s.mask

        nu = torch.as_tensor(np.asarray(flow.viscosity), dtype=torch.float32, device=device)
        nu = nu[..., None, None] if nu.dim() else nu
        linear = -nu * k2 - flow.drag
        self.explicit = (1.0 + 0.5 * dt * linear).to(torch.complex64)
        self.implicit = (1.0 / (1.0 - 0.5 * dt * linear)).to(torch.complex64)

    def to_spectral(self, w: torch.Tensor) -> torch.Tensor:
        """omega_hat of physical vorticity (..., n, n), projected onto the dealiased modes."""
        return self.spectral.forward(w.to(torch.float32).contiguous()) * self.spectral.mask

    def to_physical(self, w_hat: torch.Tensor) -> torch.Tensor:
        return self.spectral.inverse(w_hat)

    def tendency(self, w_hat: torch.Tensor) -> torch.Tensor:
        """f_hat - N_hat, N = u . grad omega, dealiased."""
        psi_hat = w_hat * self.inverse_k2
        fields_hat = torch.stack(
            [self.iky * psi_hat, -self.ikx * psi_hat, self.ikx * w_hat, self.iky * w_hat]
        )
        u, v, wx, wy = self.spectral.inverse(fields_hat)  # one batched inverse transform
        nonlinear_hat = self.spectral.forward(u * wx + v * wy) * self.spectral.mask
        return self.forcing_hat - nonlinear_hat

    def step(self, w_hat: torch.Tensor) -> torch.Tensor:
        f0 = self.tendency(w_hat)
        predicted = (self.explicit * w_hat + self.dt * f0) * self.implicit
        f1 = self.tendency(predicted)
        return (self.explicit * w_hat + (0.5 * self.dt) * (f0 + f1)) * self.implicit

    def frames(self, w0: torch.Tensor, steps: int, record_every: int = 1) -> Iterator[torch.Tensor]:
        """Yields physical vorticity (..., n, n) every record_every steps, starting with the
        dealiased initial condition, so callers can stream frames out without holding a whole
        trajectory."""
        if steps < 0 or record_every < 1:
            raise ValueError("need steps >= 0 and record_every >= 1")
        w_hat = self.to_spectral(w0)
        yield self.to_physical(w_hat)
        for k in range(1, steps + 1):
            w_hat = self.step(w_hat)
            if k % record_every == 0:
                yield self.to_physical(w_hat)

    def simulate(self, w0: torch.Tensor, steps: int, record_every: int = 1) -> torch.Tensor:
        """All frames, stacked: (frames, ..., n, n), like flow.reference.simulate."""
        return torch.stack(list(self.frames(w0, steps, record_every)))
