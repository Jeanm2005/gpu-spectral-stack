"""Checks the float64 references and the tolerance policy, before any kernel is compared
against them: the references against the DFT's definition and each other, the policy against
production float32 FFTs (pocketfft on the CPU, cuFFT on the GPU)."""

import numpy as np
import pytest
import torch

from spectral import reference as ref
from spectral.testing import (
    FLOAT32_EPS,
    POW2_SIZES,
    assert_transform_close,
    fft_tolerance,
    random_complex,
    random_real,
)

# --- The references ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 2, 3, 5, 8, 12, 16, 31, 64])
def test_fft_matches_the_definition(rng, n):
    x = random_complex(rng, (4, n))
    np.testing.assert_allclose(ref.fft(x), ref.dft(x), rtol=0, atol=1e-10 * n)


@pytest.mark.parametrize("n", [2, 64, 4096])
def test_ifft_inverts_fft(rng, n):
    x = random_complex(rng, (3, n))
    np.testing.assert_allclose(ref.ifft(ref.fft(x)), x, rtol=0, atol=1e-12)


@pytest.mark.parametrize("n", [8, 9, 64])
def test_rfft_is_the_nonnegative_half_of_fft(rng, n):
    x = random_real(rng, (3, n))
    np.testing.assert_allclose(ref.rfft(x), ref.fft(x)[..., : n // 2 + 1], rtol=0, atol=1e-10)


@pytest.mark.parametrize("n", [8, 9, 64])
def test_irfft_inverts_rfft(rng, n):
    x = random_real(rng, (3, n))
    np.testing.assert_allclose(ref.irfft(ref.rfft(x), n), x, rtol=0, atol=1e-12)


def test_fft2_is_row_transforms_then_column_transforms(rng):
    # The decomposition the 2D kernel (Task 1.6) is built on.
    x = random_complex(rng, (2, 8, 16))
    rows_then_columns = np.fft.fft(ref.fft(x), axis=-2)
    np.testing.assert_allclose(ref.fft2(x), rows_then_columns, rtol=0, atol=1e-10)


def test_rfft2_truncate_keeps_the_lowest_modes(rng):
    x = random_real(rng, (2, 16, 16))
    full = ref.rfft2(x)
    modes = ref.rfft2_truncate(x, m1=3, m2=4)
    assert modes.shape == (2, 6, 4)
    np.testing.assert_array_equal(modes[..., :3, :], full[..., :3, :4])
    np.testing.assert_array_equal(modes[..., 3:, :], full[..., 13:, :4])


def test_keeping_every_mode_round_trips(rng):
    x = random_real(rng, (2, 8, 8))
    modes = ref.rfft2_truncate(x, m1=4, m2=5)  # 2 * 4 = 8 rows and 8 // 2 + 1 = 5 columns
    np.testing.assert_allclose(ref.irfft2_from_modes(modes, (8, 8)), x, rtol=0, atol=1e-12)


def test_truncation_is_a_low_pass_filter():
    # A low and a high spatial frequency; keeping 4 x 4 modes must return only the low one.
    i, j = np.meshgrid(np.arange(32), np.arange(32), indexing="ij")
    low = np.cos(2 * np.pi * (1 * i + 2 * j) / 32)
    high = np.cos(2 * np.pi * (10 * i + 12 * j) / 32)
    modes = ref.rfft2_truncate(low + high, m1=4, m2=4)
    np.testing.assert_allclose(ref.irfft2_from_modes(modes, (32, 32)), low, rtol=0, atol=1e-12)


# --- The tolerance policy ---------------------------------------------------------------------


@pytest.mark.parametrize("n", POW2_SIZES)
def test_policy_accepts_float32_rounding_of_the_exact_result(rng, n):
    x = random_complex(rng, (8, n))
    exact = ref.fft(x)
    assert_transform_close(exact.astype(np.complex64), exact, n)


def test_policy_rejects_one_bad_transform_in_a_batch(rng):
    n = 1024
    x = random_complex(rng, (64, n))
    exact = ref.fft(x)
    actual = exact.copy()
    actual[17, 5] += 10 * fft_tolerance(n) * np.linalg.norm(exact[17])
    with pytest.raises(AssertionError, match=r"exceeds the tolerance .* in transform \(17,\)"):
        assert_transform_close(actual, exact, n)


def test_policy_limit_is_eight_eps_per_stage():
    assert fft_tolerance(4096) == pytest.approx(8 * FLOAT32_EPS * 12)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_torch_fft_meets_the_policy(rng, device):
    # A production float32 FFT passes at every size, so the limit is attainable.
    for n in POW2_SIZES:
        x = random_complex(rng, (64, n))
        assert_transform_close(torch.fft.fft(torch.from_numpy(x).to(device)), ref.fft(x), n)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_torch_rfft2_meets_the_policy(rng, device):
    for n1, n2 in [(64, 64), (128, 256)]:
        x = random_real(rng, (8, n1, n2))
        actual = torch.fft.rfft2(torch.from_numpy(x).to(device))
        assert_transform_close(actual, ref.rfft2(x), n1 * n2, axes=(-2, -1))
