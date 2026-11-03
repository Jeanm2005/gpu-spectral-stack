"""Correctness of our FFT implementations against the float64 reference, with the tolerance
policy of spectral.testing. Every implementation in spectral.FFT_IMPLS runs every test."""

import numpy as np
import pytest
import torch

import spectral
from spectral import reference as ref
from spectral.testing import (
    BATCHES,
    POW2_SIZES,
    assert_transform_close,
    fft_tolerance,
    random_complex,
)

pytestmark = pytest.mark.gpu  # every test here runs our kernels

IMPLS = spectral.FFT_IMPLS
SIZES = [1, *POW2_SIZES]


def run(x: np.ndarray, *, inverse: bool = False, impl: str) -> np.ndarray:
    y = spectral.fft_c2c(torch.from_numpy(x).cuda(), inverse=inverse, impl=impl)
    return y.cpu().numpy()


# --- Against the reference, at every size and batch ------------------------------------------


@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("batch", BATCHES)
@pytest.mark.parametrize("n", SIZES)
def test_forward_matches_the_reference(rng, impl, n, batch):
    x = random_complex(rng, (batch, n))
    assert_transform_close(run(x, impl=impl), ref.fft(x), n)


@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("batch", BATCHES)
@pytest.mark.parametrize("n", SIZES)
def test_inverse_matches_the_reference(rng, impl, n, batch):
    x = random_complex(rng, (batch, n))
    assert_transform_close(run(x, inverse=True, impl=impl), ref.ifft(x), n)


@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("n", [2, 64, 4096])
def test_inverse_undoes_forward(rng, impl, n):
    x = torch.from_numpy(random_complex(rng, (3, n))).cuda()
    back = spectral.fft_c2c(spectral.fft_c2c(x, impl=impl), inverse=True, impl=impl)
    assert_transform_close(back, x, n, tol=2 * fft_tolerance(n))  # two transforms' worth


# --- Known answers ----------------------------------------------------------------------------


@pytest.mark.parametrize("impl", IMPLS)
def test_impulse_transforms_to_all_ones(impl):
    x = np.zeros((1, 256), dtype=np.complex64)
    x[0, 0] = 1
    np.testing.assert_allclose(run(x, impl=impl), np.ones((1, 256)), rtol=0, atol=1e-6)


@pytest.mark.parametrize("impl", IMPLS)
def test_single_frequency_lands_in_one_bin(impl):
    n, k0 = 256, 5
    m = np.arange(n)
    x = np.exp(2j * np.pi * k0 * m / n).astype(np.complex64)[None, :]
    expected = np.zeros((1, n), dtype=np.complex128)
    expected[0, k0] = n
    assert_transform_close(run(x, impl=impl), expected, n)


# --- Shapes and layouts -----------------------------------------------------------------------


@pytest.mark.parametrize("impl", IMPLS)
def test_leading_dimensions_are_batch_dimensions(rng, impl):
    x = random_complex(rng, (2, 3, 64))
    y = run(x, impl=impl)
    assert y.shape == (2, 3, 64)
    assert_transform_close(y, ref.fft(x), 64)


@pytest.mark.parametrize("impl", IMPLS)
def test_non_contiguous_input(rng, impl):
    x = torch.from_numpy(random_complex(rng, (64, 128))).cuda().T  # shape (128, 64), strided
    assert not x.is_contiguous()
    assert_transform_close(spectral.fft_c2c(x, impl=impl), ref.fft(x.cpu().numpy()), 64)


@pytest.mark.parametrize("impl", IMPLS)
def test_empty_input_gives_empty_output(impl):
    x = torch.zeros(0, 8, dtype=torch.complex64, device="cuda")
    assert spectral.fft_c2c(x, impl=impl).shape == (0, 8)


# --- Rejected inputs --------------------------------------------------------------------------


@pytest.mark.parametrize("impl", IMPLS)
def test_rejects_a_size_that_is_not_a_power_of_two(impl):
    x = torch.zeros(4, 12, dtype=torch.complex64, device="cuda")
    with pytest.raises(RuntimeError, match="power of two"):
        spectral.fft_c2c(x, impl=impl)


def test_rejects_an_unknown_implementation():
    x = torch.zeros(4, 8, dtype=torch.complex64, device="cuda")
    with pytest.raises(RuntimeError, match="unknown FFT implementation"):
        spectral.fft_c2c(x, impl="nope")
