"""gpu-spectral-stack: FFT kernels written in CUDA, and the layers built on them."""

import torch  # noqa: F401  # loads libtorch before the extension, which links against it

from . import _C  # noqa: F401  # registers the torch.ops.spectral operators
from .ops import scale

__all__ = ["scale"]
