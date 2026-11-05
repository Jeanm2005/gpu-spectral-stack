"""Real-to-complex and complex-to-real transforms (Task 1.5) against the float64 reference, for
every implementation in spectral.FFT_IMPLS: the generic path (an n/2-point complex transform plus a
combine pass) and the fused Stockham kernels."""

import numpy as np
import pytest
import torch

import spectral
from spectral import reference as ref
from spectral.testing import (
    BATCHES,
    assert_transform_close,
    fft_tolerance,
    random_complex,
    random_real,
)

pytestmark = pytest.mark.gpu

IMPLS = spectral.FFT_IMPLS
# A real transform of n points runs one n/2-point complex transform, so the single-block kernels
# reach n = 8192.
SIZES = [2**p for p in range(1, 14)]
SIZE_LIMITED = [impl for impl in IMPLS if impl.startswith("stockham")]


def rfft(x: np.ndarray, impl: str) -> np.ndarray:
    return spectral.rfft(torch.from_numpy(x).cuda(), impl=impl).cpu().numpy()


def irfft(x: np.ndarray, n: int, impl: str) -> np.ndarray:
    return spectral.irfft(torch.from_numpy(x).cuda(), n, impl=impl).cpu().numpy()


def hermitian_spectrum(rng, batch: int, n: int) -> np.ndarray:
    """The rfft of a random real signal: a valid input for irfft."""
    return ref.rfft(random_real(rng, (batch, n))).astype(np.complex64)


@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("batch", BATCHES)
@pytest.mark.parametrize("n", SIZES)
def test_rfft_matches_the_reference(rng, impl, n, batch):
    x = random_real(rng, (batch, n))
    y = rfft(x, impl)
    assert y.shape == (batch, n // 2 + 1)
    assert_transform_close(y, ref.rfft(x), n)


@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("batch", BATCHES)
@pytest.mark.parametrize("n", SIZES)
def test_irfft_matches_the_reference(rng, impl, n, batch):
    spectrum = hermitian_spectrum(rng, batch, n)
    y = irfft(spectrum, n, impl)
    assert y.shape == (batch, n)
    assert y.dtype == np.float32
    assert_transform_close(y, ref.irfft(spectrum, n), n)


@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("n", [2, 64, 8192])
def test_irfft_undoes_rfft(rng, impl, n):
    x = torch.from_numpy(random_real(rng, (3, n))).cuda()
    back = spectral.irfft(spectral.rfft(x, impl=impl), n, impl=impl)
    assert_transform_close(back, x, n, tol=2 * fft_tolerance(n))


@pytest.mark.parametrize("impl", IMPLS)
def test_irfft_of_a_non_hermitian_spectrum_matches_torch(rng, impl):
    # Arbitrary complex input, imaginary parts at frequency 0 and n/2 included: torch (like
    # numpy) ignores those two, and so must we.
    n = 256
    spectrum = random_complex(rng, (4, n // 2 + 1))
    expected = torch.fft.irfft(torch.from_numpy(spectrum).to(torch.complex128), n).numpy()
    assert_transform_close(irfft(spectrum, n, impl), expected, n)


@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("batch", [255, 257, 1000])
def test_batches_that_do_not_fill_the_last_block(rng, impl, batch):
    x = random_real(rng, (batch, 16))
    assert_transform_close(rfft(x, impl), ref.rfft(x), 16)
    spectrum = hermitian_spectrum(rng, batch, 16)
    assert_transform_close(irfft(spectrum, 16, impl), ref.irfft(spectrum, 16), 16)


@pytest.mark.parametrize("impl", IMPLS)
def test_constant_signal_lands_in_bin_zero(impl):
    x = np.ones((1, 128), dtype=np.float32)
    expected = np.zeros((1, 65), dtype=np.complex128)
    expected[0, 0] = 128
    np.testing.assert_allclose(rfft(x, impl), expected, rtol=0, atol=1e-4)


@pytest.mark.parametrize("impl", IMPLS)
def test_leading_dimensions_and_non_contiguous_input(rng, impl):
    x = torch.from_numpy(random_real(rng, (64, 2, 3))).cuda().permute(1, 2, 0)  # (2, 3, 64)
    assert not x.is_contiguous()
    y = spectral.rfft(x, impl=impl)
    assert y.shape == (2, 3, 33)
    assert_transform_close(y, ref.rfft(x.cpu().numpy()), 64)
    back = spectral.irfft(y.transpose(0, 1), 64, impl=impl)  # non-contiguous complex input
    assert back.shape == (3, 2, 64)
    assert_transform_close(back, x.transpose(0, 1), 64, tol=2 * fft_tolerance(64))


@pytest.mark.parametrize("impl", IMPLS)
def test_empty_input_gives_empty_output(impl):
    x = torch.zeros(0, 8, device="cuda")
    assert spectral.rfft(x, impl=impl).shape == (0, 5)
    empty_spectrum = torch.zeros(0, 5, dtype=torch.complex64, device="cuda")
    assert spectral.irfft(empty_spectrum, 8, impl=impl).shape == (0, 8)


# --- Rejected inputs --------------------------------------------------------------------------


@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("n", [1, 12])
def test_rejects_sizes_that_are_not_a_power_of_two_from_2(impl, n):
    with pytest.raises(RuntimeError, match="power of two of at least 2"):
        spectral.rfft(torch.zeros(2, n, device="cuda"), impl=impl)


@pytest.mark.parametrize("impl", SIZE_LIMITED)
def test_rejects_a_size_above_the_implementation_limit(impl):
    with pytest.raises(RuntimeError, match="real sizes up to 8192"):
        spectral.rfft(torch.zeros(2, 16384, device="cuda"), impl=impl)


def test_rejects_a_spectrum_of_the_wrong_length():
    x = torch.zeros(2, 8, dtype=torch.complex64, device="cuda")
    with pytest.raises(RuntimeError, match="n/2 \\+ 1 = 5 inputs"):
        spectral.irfft(x, 8)


def test_rejects_wrong_dtypes():
    with pytest.raises(RuntimeError, match="float32"):
        spectral.rfft(torch.zeros(2, 8, dtype=torch.complex64, device="cuda"))
    with pytest.raises(RuntimeError, match="complex64"):
        spectral.irfft(torch.zeros(2, 5, device="cuda"), 8)
