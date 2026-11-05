"""gpu-spectral-stack: FFT kernels written in CUDA, and the layers built on them."""

import torch  # noqa: F401  # loads libtorch before the extension, which links against it

from . import _C  # noqa: F401  # registers the torch.ops.spectral operators
from .ops import (
    FFT_IMPLS,
    fft2,
    fft_c2c,
    fft_truncated,
    ifft_from_modes,
    irfft,
    irfft2,
    irfft2_from_modes,
    rfft,
    rfft2,
    rfft2_truncated,
    scale,
)

__all__ = [
    "FFT_IMPLS",
    "fft2",
    "fft_c2c",
    "fft_truncated",
    "ifft_from_modes",
    "irfft",
    "irfft2",
    "irfft2_from_modes",
    "rfft",
    "rfft2",
    "rfft2_truncated",
    "scale",
]
