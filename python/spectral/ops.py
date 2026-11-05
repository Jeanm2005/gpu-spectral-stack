"""Python entry points for the torch.ops.spectral operators."""

import torch

from . import _C

# Every FFT implementation the extension provides, in the order they were written. Read from the
# C++ table (csrc/fft/registry.hpp), so adding an implementation there adds it to every test.
FFT_IMPLS: tuple[str, ...] = _C.fft_impls()


def scale(x: torch.Tensor, alpha: float) -> torch.Tensor:
    """Returns alpha * x, computed by our own CUDA kernel.

    x must be a complex64 tensor on a CUDA device. Non-contiguous inputs are copied to a
    contiguous layout first. The result is a new contiguous tensor.
    """
    return torch.ops.spectral.scale.default(x, alpha)


def fft_c2c(x: torch.Tensor, *, inverse: bool = False, impl: str = "naive") -> torch.Tensor:
    """Batched complex FFT along the last dimension, computed by one of our CUDA kernels.

    x must be complex64 on a CUDA device with a power-of-two last dimension; every leading
    dimension is a batch dimension. The forward transform is unnormalized and the inverse
    divides by n, as torch.fft.fft and torch.fft.ifft do. Returns a new contiguous tensor.
    `impl` is one of FFT_IMPLS; some implementations have a largest supported size.
    """
    return torch.ops.spectral.fft_c2c.default(x, inverse, impl)
