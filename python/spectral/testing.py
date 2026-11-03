"""Comparing our transforms against the float64 references: the tolerance policy and helpers.

Tolerance policy
----------------
Each FFT stage rounds every value once, so float32 error grows with the number of stages,
log2 N: the worst case grows like eps * log2 N, the typical (root-mean-square) case like
eps * sqrt(log2 N). We allow

    relative L2 error  <=  FFT_TOLERANCE_FACTOR * eps * log2 N      (N = N1 * N2 in 2D)

per transform. The factor 8 leaves room for twiddle factors from fast sine/cosine intrinsics,
whose error adds roughly one more eps per stage. Real bugs (a wrong index, a missing twiddle,
a race) produce errors of order 1, many orders of magnitude above the limit.

The error is normwise (||actual - expected|| / ||expected||) because bins close to zero have
meaningless elementwise relative errors. It is measured per transform, so one bad transform
cannot hide among sixty-three good ones in a batch.
"""

from __future__ import annotations

import math

import numpy as np

FLOAT32_EPS = float(np.finfo(np.float32).eps)  # 2^-23, about 1.19e-7
FFT_TOLERANCE_FACTOR = 8.0

POW2_SIZES = [2**p for p in range(1, 13)]  # 2 .. 4096, the single-block kernel range
BATCHES = [1, 3, 64]


def fft_tolerance(n: int, eps: float = FLOAT32_EPS) -> float:
    """The largest relative L2 error allowed for one transform of n points."""
    return FFT_TOLERANCE_FACTOR * eps * max(1.0, math.log2(n))


def to_numpy(x) -> np.ndarray:
    """A NumPy array from a NumPy array or a torch.Tensor on any device."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def relative_error(actual, expected, axes: tuple[int, ...]) -> np.ndarray:
    """||actual - expected|| / ||expected|| for each transform, the norms taken over `axes`."""
    a = to_numpy(actual).astype(np.complex128)
    e = to_numpy(expected).astype(np.complex128)
    if a.shape != e.shape:
        raise AssertionError(f"shape {a.shape} differs from the expected {e.shape}")
    num = np.sqrt(np.sum(np.abs(a - e) ** 2, axis=axes))
    den = np.sqrt(np.sum(np.abs(e) ** 2, axis=axes))
    return num / np.maximum(den, np.finfo(np.float64).tiny)


def assert_transform_close(
    actual, expected, n: int, axes: tuple[int, ...] = (-1,), tol: float | None = None
) -> None:
    """Fails if any transform in the batch exceeds the tolerance for n points."""
    tol = fft_tolerance(n) if tol is None else tol
    err = relative_error(actual, expected, axes)
    worst = float(np.max(err))
    if worst > tol:
        where = np.unravel_index(int(np.argmax(err)), np.shape(err)) if np.ndim(err) else ()
        raise AssertionError(
            f"relative L2 error {worst:.3e} exceeds the tolerance {tol:.3e} "
            f"(n = {n}) in transform {tuple(int(i) for i in where)}"
        )


def random_complex(rng: np.random.Generator, shape, dtype=np.complex64) -> np.ndarray:
    """Complex Gaussian samples with unit variance (half in the real part, half imaginary)."""
    z = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    return (z / math.sqrt(2.0)).astype(dtype)


def random_real(rng: np.random.Generator, shape, dtype=np.float32) -> np.ndarray:
    return rng.standard_normal(shape).astype(dtype)
