"""gpu-spectral-stack: FFT kernels written in CUDA, and the layers built on them."""

import torch  # noqa: F401  # loads libtorch before the extension, which links against it

from . import _C  # noqa: F401  # registers the torch.ops.spectral operators
from .ops import FFT_IMPLS, fft2, fft_c2c, irfft, irfft2, rfft, rfft2, scale

__all__ = ["FFT_IMPLS", "fft2", "fft_c2c", "irfft", "irfft2", "rfft", "rfft2", "scale"]
