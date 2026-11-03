"""Float64 NumPy references for every transform this repository implements.

Conventions (the same as numpy.fft and torch.fft with norm="backward"):
- Transforms act on the last axis (1D) or the last two axes (2D); leading axes are batch.
- Forward transforms are unnormalized: X[k] = sum_n x[n] exp(-2 pi i k n / N).
- Inverse transforms divide by N (by N1 * N2 in 2D), so ifft(fft(x)) == x.
- Real-input transforms keep the N // 2 + 1 non-negative frequencies of the last axis.
- Truncated transforms keep only the low-frequency modes an FNO spectral convolution uses.

Every function computes in complex128 / float64, whatever the input precision, so a float32
result can be compared against it. Pass the reference the same float32 input the kernel saw:
the comparison then measures the transform's error alone, not the rounding of the input.
"""

from __future__ import annotations

import numpy as np


def _c128(x) -> np.ndarray:
    return np.asarray(x, dtype=np.complex128)


def _f64(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def dft(x) -> np.ndarray:
    """The discrete Fourier transform by its definition, O(N^2). Checks the references below."""
    x = _c128(x)
    n = x.shape[-1]
    k = np.arange(n)
    w = np.exp(-2j * np.pi * np.outer(k, k) / n)  # w[k, m] = exp(-2 pi i k m / n), symmetric
    return x @ w


def fft(x) -> np.ndarray:
    return np.fft.fft(_c128(x), axis=-1)


def ifft(x) -> np.ndarray:
    return np.fft.ifft(_c128(x), axis=-1)


def rfft(x) -> np.ndarray:
    return np.fft.rfft(_f64(x), axis=-1)


def irfft(x, n: int) -> np.ndarray:
    """Inverse of rfft for an output of n points. Like numpy and torch, the imaginary parts of
    the zero-frequency bin (and of the Nyquist bin when n is even) are ignored."""
    return np.fft.irfft(_c128(x), n=n, axis=-1)


def fft2(x) -> np.ndarray:
    return np.fft.fft2(_c128(x), axes=(-2, -1))


def ifft2(x) -> np.ndarray:
    return np.fft.ifft2(_c128(x), axes=(-2, -1))


def rfft2(x) -> np.ndarray:
    return np.fft.rfft2(_f64(x), axes=(-2, -1))


def irfft2(x, shape: tuple[int, int]) -> np.ndarray:
    return np.fft.irfft2(_c128(x), s=shape, axes=(-2, -1))


def fft_truncate(x, k: int) -> np.ndarray:
    """The first k outputs of fft(x): what a pruned 1D forward transform computes."""
    n = np.shape(x)[-1]
    if not 0 < k <= n:
        raise ValueError(f"k must be in [1, {n}], got {k}")
    return fft(x)[..., :k]


def rfft2_truncate(x, m1: int, m2: int) -> np.ndarray:
    """The low-frequency modes of rfft2(x) that an FNO spectral convolution keeps.

    Along the second-to-last axis (length N1, full spectrum) the lowest frequencies are rows
    0 .. m1-1 (positive) and N1-m1 .. N1-1 (negative). Along the last axis (N2 // 2 + 1
    non-negative frequencies) they are columns 0 .. m2-1. The result has shape (..., 2*m1, m2):
    the positive-frequency rows first, then the negative-frequency rows.
    """
    n1, n2 = np.shape(x)[-2:]
    if not (0 < 2 * m1 <= n1 and 0 < m2 <= n2 // 2 + 1):
        raise ValueError(f"need 0 < 2*m1 <= {n1} and 0 < m2 <= {n2 // 2 + 1}, got {m1}, {m2}")
    full = rfft2(x)
    return np.concatenate([full[..., :m1, :m2], full[..., n1 - m1 :, :m2]], axis=-2)


def irfft2_from_modes(modes, shape: tuple[int, int]) -> np.ndarray:
    """Inverse of rfft2_truncate: zero-fills every dropped mode, then applies irfft2."""
    modes = _c128(modes)
    n1, n2 = shape
    two_m1, m2 = modes.shape[-2:]
    m1 = two_m1 // 2
    if two_m1 % 2 or not (0 < two_m1 <= n1 and 0 < m2 <= n2 // 2 + 1):
        raise ValueError(f"modes of shape {modes.shape[-2:]} do not fit an output of {shape}")
    full = np.zeros(modes.shape[:-2] + (n1, n2 // 2 + 1), dtype=np.complex128)
    full[..., :m1, :m2] = modes[..., :m1, :]
    full[..., n1 - m1 :, :m2] = modes[..., m1:, :]
    return irfft2(full, shape)
