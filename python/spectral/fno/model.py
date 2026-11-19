"""The Fourier Neural Operator for the Phase 2 flow data (Tasks 3.2 and 3.3).

The 2D time-stepping FNO of Li et al., "Fourier Neural Operator for Parametric Partial
Differential Equations" (ICLR 2021): from the last `in_frames` vorticity frames, predict the next
`out_frames`.

    input     (B, in_frames, n, n) vorticity, plus log-viscosity and the (y, x) grid as three
              more channels: (B, in_frames + 3, n, n)
    lift      1x1 convolution to `width` channels
    layers    v <- gelu(SpectralConv2d(v) + Conv1x1(v)), without the gelu after the last layer
    project   1x1 convolutions width -> projection -> out_frames

A spectral convolution keeps only the lowest modes of rfft2(v): rows 0 .. m1-1 and n1-m1 .. n1-1
(non-negative and negative frequencies along y) and columns 0 .. m2-1 (along x). That is the
(2 m1, m2) layout of spectral.reference.rfft2_truncate and of spectral.rfft2_truncated, so the
same weights work with every backend. Each kept mode is mixed across channels by its own learned
complex C_in x C_out matrix; every other mode is set to zero.

The convolution itself is a plain function chosen by name (SPECTRAL_CONV_BACKENDS), so the
layer can switch from torch.fft to our own kernels (Task 3.4) without changing the model.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn


def spectral_conv2d_torch(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """The reference spectral convolution, on torch.fft.

    x: (B, C_in, n1, n2) real. weight: (C_in, C_out, 2 m1, m2) complex, is the truncated layout
    Returns (B, C_out, n1, n2) real.
    """
    n1, n2 = x.shape[-2:]
    _, c_out, two_m1, m2 = weight.shape
    m1 = two_m1 // 2

    x_ft = torch.fft.rfft2(x)  # (B, C_in, n1, n2 // 2 + 1)
    modes = torch.cat([x_ft[..., :m1, :m2], x_ft[..., n1 - m1 :, :m2]], dim=-2)
    mixed = torch.einsum("bixy,ioxy->boxy", modes, weight)

    out_ft = x_ft.new_zeros(x.shape[0], c_out, n1, n2 // 2 + 1)
    out_ft[..., :m1, :m2] = mixed[..., :m1, :]
    out_ft[..., n1 - m1 :, :m2] = mixed[..., m1:, :]
    return torch.fft.irfft2(out_ft, s=(n1, n2))


# Every implementation of the spectral convolution, by name.
SPECTRAL_CONV_BACKENDS: dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
    "torch": spectral_conv2d_torch,
}


class SpectralConv2d(nn.Module):
    def __init__(self, c_in: int, c_out: int, m1: int, m2: int, backend: str = "torch"):
        super().__init__()
        if backend not in SPECTRAL_CONV_BACKENDS:
            raise ValueError(
                f"unknown backend {backend!r}; available: {sorted(SPECTRAL_CONV_BACKENDS)}"
            )
        if m1 < 1 or m2 < 1:
            raise ValueError(f"need m1, m2 >= 1, got {m1}, {m2}")
        self.m1, self.m2, self.backend = m1, m2, backend
        scale = 1.0 / (c_in * c_out)
        self.weight = nn.Parameter(scale * torch.rand(c_in, c_out, 2 * m1, m2, dtype=torch.cfloat))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n1, n2 = x.shape[-2:]
        if 2 * self.m1 > n1 or self.m2 > n2 // 2 + 1:
            raise ValueError(
                f"{self.m1} x {self.m2} modes do not fit a {n1} x {n2} grid: "
                f"need 2 m1 <= {n1} and m2 <= {n2 // 2 + 1}"
            )
        return SPECTRAL_CONV_BACKENDS[self.backend](x, self.weight)

    def extra_repr(self) -> str:
        c_in, c_out = self.weight.shape[:2]
        return f"{c_in}, {c_out}, modes=({self.m1}, {self.m2}), backend={self.backend}"


class FNO2d(nn.Module):
    def __init__(
        self,
        in_frames: int,
        out_frames: int,
        *,
        modes: int = 12,
        width: int = 32,
        layers: int = 4,
        projection: int = 128,
        backend: str = "torch",
    ):
        super().__init__()
        if layers < 1:
            raise ValueError(f"need at least one Fourier layer, got {layers}")
        self.lift = nn.Conv2d(in_frames + 3, width, 1)
        self.spectral = nn.ModuleList(
            SpectralConv2d(width, width, modes, modes, backend) for _ in range(layers)
        )
        self.pointwise = nn.ModuleList(nn.Conv2d(width, width, 1) for _ in range(layers))
        self.project = nn.Sequential(
            nn.Conv2d(width, projection, 1), nn.GELU(), nn.Conv2d(projection, out_frames, 1)
        )

    def forward(self, frames: torch.Tensor, log_viscosity: torch.Tensor) -> torch.Tensor:
        """frames: (B, in_frames, n1, n2); log_viscosity: (B,), already normalized."""
        b, _, n1, n2 = frames.shape
        y = torch.arange(n1, device=frames.device, dtype=frames.dtype) / n1
        x = torch.arange(n2, device=frames.device, dtype=frames.dtype) / n2
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        extra = torch.stack(
            [
                log_viscosity.to(frames.dtype).view(b, 1, 1).expand(b, n1, n2),
                grid_y.expand(b, n1, n2),
                grid_x.expand(b, n1, n2),
            ],
            dim=1,
        )
        v = self.lift(torch.cat([frames, extra], dim=1))
        last = len(self.spectral) - 1
        for i, (spectral, pointwise) in enumerate(zip(self.spectral, self.pointwise, strict=True)):
            v = spectral(v) + pointwise(v)
            if i < last:
                v = F.gelu(v)
        return self.project(v)


def relative_l2(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """||prediction - target|| / ||target|| for each item of the batch (first dimension): the
    standard FNO training loss and test metric."""
    diff = (prediction - target).flatten(1).norm(dim=1)
    return diff / target.flatten(1).norm(dim=1).clamp_min(torch.finfo(target.dtype).tiny)
