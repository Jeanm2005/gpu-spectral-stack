"""gpu-spectral-stack: FFT kernels written in CUDA, and the layers built on them."""

import torch  # noqa: F401  # loads libtorch before the extension, which links against it

from . import _C  # noqa: F401  # registers the torch.ops.spectral operators
from .ops import FFT_IMPLS, fft_c2c, irfft, rfft, scale

__all__ = ["FFT_IMPLS", "fft_c2c", "irfft", "rfft", "scale"]
