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


def rfft(x: torch.Tensor, *, impl: str = "naive") -> torch.Tensor:
    """Batched real-to-complex FFT along the last dimension, like torch.fft.rfft.

    x must be float32 on a CUDA device with a power-of-two last dimension n >= 2. Returns a new
    complex64 tensor whose last dimension holds the n // 2 + 1 non-negative frequencies.
    Computed as one n/2-point complex transform (see csrc/fft/real.cuh).
    """
    return torch.ops.spectral.fft_r2c.default(x, impl)


def irfft(x: torch.Tensor, n: int, *, impl: str = "naive") -> torch.Tensor:
    """Inverse of rfft, divided by n, like torch.fft.irfft(x, n).

    x must be complex64 on a CUDA device with n // 2 + 1 values in its last dimension, n a power
    of two >= 2. The imaginary parts of the zero and n/2 frequencies are ignored, as in torch.
    Returns a new float32 tensor with n values in its last dimension.
    """
    return torch.ops.spectral.fft_c2r.default(x, n, impl)


# 2D transforms (Task 1.6). `columns` picks how the column transforms run: "strided" transforms
# them in place with a kernel that reads columns directly (the Stockham implementations have
# one), "transpose" transposes, runs row transforms, and transposes back (every implementation),
# and "auto" takes "strided" when it exists.


def fft2(
    x: torch.Tensor, *, inverse: bool = False, impl: str = "naive", columns: str = "auto"
) -> torch.Tensor:
    """Batched 2D complex FFT over the last two dimensions, like torch.fft.fft2 (or ifft2).

    x must be complex64 on a CUDA device; both of its last two dimensions powers of two.
    """
    return torch.ops.spectral.fft2_c2c.default(x, inverse, impl, columns)


def rfft2(x: torch.Tensor, *, impl: str = "naive", columns: str = "auto") -> torch.Tensor:
    """Batched 2D real-to-complex FFT over the last two dimensions, like torch.fft.rfft2.

    x must be float32 on a CUDA device, (..., n1, n2) with powers of two and n2 >= 2. Returns
    complex64 of shape (..., n1, n2 // 2 + 1).
    """
    return torch.ops.spectral.fft2_r2c.default(x, impl, columns)


def irfft2(x: torch.Tensor, n2: int, *, impl: str = "naive", columns: str = "auto") -> torch.Tensor:
    """Inverse of rfft2, like torch.fft.irfft2(x, s=(x.shape[-2], n2)): divides by n1 * n2.

    x must be complex64 of shape (..., n1, n2 // 2 + 1) on a CUDA device. Returns float32 of
    shape (..., n1, n2).
    """
    return torch.ops.spectral.fft2_c2r.default(x, n2, impl, columns)
