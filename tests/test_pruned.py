"""Pruned transforms (Task 1.7) against the float64 reference, for every implementation: the
Stockham kernels prune inside the kernel, the others gather from a full transform."""

import numpy as np
import pytest
import torch

import spectral
from spectral import reference as ref
from spectral.testing import assert_transform_close, fft_tolerance, random_complex, random_real

pytestmark = pytest.mark.gpu

IMPLS = spectral.FFT_IMPLS
AXES = (-2, -1)


def cuda(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(x).cuda()


def ifft_from_modes_reference(modes: np.ndarray, n: int) -> np.ndarray:
    full = np.zeros(modes.shape[:-1] + (n,), dtype=np.complex128)
    full[..., : modes.shape[-1]] = modes
    return ref.ifft(full)


# --- 1D complex -------------------------------------------------------------------------------


@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize(("n", "k"), [(1, 1), (8, 1), (8, 3), (64, 16), (256, 255), (4096, 32)])
@pytest.mark.parametrize("batch", [1, 3, 300])
def test_fft_truncated_matches_the_reference(rng, impl, n, k, batch):
    x = random_complex(rng, (batch, n))
    y = spectral.fft_truncated(cuda(x), k, impl=impl)
    assert y.shape == (batch, k)
    # The error is relative to the kept modes' norm, so the full transform's tolerance applies.
    assert_transform_close(y, ref.fft_truncate(x, k), n)


@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize(("n", "k"), [(1, 1), (8, 1), (8, 3), (64, 16), (256, 255), (4096, 32)])
@pytest.mark.parametrize("batch", [1, 3, 300])
def test_ifft_from_modes_matches_the_reference(rng, impl, n, k, batch):
    modes = random_complex(rng, (batch, k))
    y = spectral.ifft_from_modes(cuda(modes), n, impl=impl)
    assert y.shape == (batch, n)
    assert_transform_close(y, ifft_from_modes_reference(modes, n), n)


@pytest.mark.parametrize("impl", IMPLS)
def test_keeping_every_mode_is_the_full_transform(rng, impl):
    x = cuda(random_complex(rng, (5, 128)))
    full = spectral.fft_c2c(x, impl=impl)
    assert torch.equal(spectral.fft_truncated(x, 128, impl=impl), full)


# --- 2D real, the FNO layout ------------------------------------------------------------------

# (n1, n2, m1, m2): square and not, the smallest, every row kept (2 m1 = n1), the Nyquist column
# kept (m2 = n2/2 + 1), and the largest single-block sizes.
CASES_2D = [
    (2, 2, 1, 1),
    (8, 8, 2, 3),
    (16, 16, 8, 9),
    (64, 32, 12, 12),
    (32, 128, 4, 20),
    (256, 256, 16, 16),
    (4096, 2, 2, 2),
    (2, 8192, 1, 64),
]


@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize(("n1", "n2", "m1", "m2"), CASES_2D)
def test_rfft2_truncated_matches_the_reference(rng, impl, n1, n2, m1, m2):
    x = random_real(rng, (3, n1, n2))
    y = spectral.rfft2_truncated(cuda(x), m1, m2, impl=impl)
    assert y.shape == (3, 2 * m1, m2)
    assert_transform_close(y, ref.rfft2_truncate(x, m1, m2), n1 * n2, axes=AXES)


@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize(("n1", "n2", "m1", "m2"), CASES_2D)
def test_irfft2_from_modes_matches_the_reference(rng, impl, n1, n2, m1, m2):
    modes = random_complex(rng, (3, 2 * m1, m2))
    y = spectral.irfft2_from_modes(cuda(modes), (n1, n2), impl=impl)
    assert y.shape == (3, n1, n2)
    assert_transform_close(y, ref.irfft2_from_modes(modes, (n1, n2)), n1 * n2, axes=AXES)


@pytest.mark.parametrize("impl", IMPLS)
def test_a_band_limited_field_survives_the_round_trip(rng, impl):
    # A field with no energy outside the kept modes is reproduced exactly (up to rounding).
    # The kept rows are frequencies 0 .. m1-1 and -m1 .. -1: in column 0, a real field's mode at
    # -m1 is the conjugate of the one at +m1, which is dropped, so the mode at -m1 must be zero
    # for the field to be band-limited.
    n1, n2, m1, m2 = 64, 64, 8, 8
    modes = ref.rfft2_truncate(random_real(rng, (2, n1, n2)), m1, m2)
    modes[..., m1, 0] = 0
    field = ref.irfft2_from_modes(modes, (n1, n2)).astype(np.float32)
    back = spectral.irfft2_from_modes(
        spectral.rfft2_truncated(cuda(field), m1, m2, impl=impl), (n1, n2), impl=impl
    )
    assert_transform_close(back, field, n1 * n2, axes=AXES, tol=2 * fft_tolerance(n1 * n2))


@pytest.mark.parametrize("impl", IMPLS)
def test_many_arrays_and_leading_dimensions(rng, impl):
    x = random_real(rng, (2, 70, 16, 32))  # 140 arrays: column blocks cross arrays
    y = spectral.rfft2_truncated(cuda(x), 4, 5, impl=impl)
    assert y.shape == (2, 70, 8, 5)
    assert_transform_close(y, ref.rfft2_truncate(x, 4, 5), 512, axes=AXES)


def test_every_implementation_agrees(rng):
    x = cuda(random_real(rng, (4, 128, 128)))
    results = [spectral.rfft2_truncated(x, 12, 12, impl=impl) for impl in IMPLS]
    for r in results[1:]:
        assert_transform_close(r, results[0], 128 * 128, axes=AXES, tol=2 * fft_tolerance(2**14))


# --- Rejected inputs --------------------------------------------------------------------------


def test_rejects_modes_that_do_not_fit():
    x = torch.zeros(8, 8, device="cuda")
    with pytest.raises(RuntimeError, match="1 <= m1 <= n1/2"):
        spectral.rfft2_truncated(x, 5, 2)
    with pytest.raises(RuntimeError, match="1 <= m1 <= n1/2"):
        spectral.rfft2_truncated(x, 2, 6)
    with pytest.raises(RuntimeError, match="do not fit"):
        spectral.irfft2_from_modes(torch.zeros(3, 2, dtype=torch.complex64, device="cuda"), (8, 8))


def test_rejects_keep_out_of_range():
    x = torch.zeros(2, 8, dtype=torch.complex64, device="cuda")
    with pytest.raises(RuntimeError, match="1 <= keep <= n"):
        spectral.fft_truncated(x, 9)
    with pytest.raises(RuntimeError, match="1 <= keep <= n"):
        spectral.ifft_from_modes(x, 4)
