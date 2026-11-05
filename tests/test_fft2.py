"""2D transforms (Task 1.6) against the float64 reference: every implementation with the transpose
method, and the ones with a strided column kernel with that method too."""

import numpy as np
import pytest
import torch

import spectral
from spectral import reference as ref
from spectral.testing import assert_transform_close, fft_tolerance, random_complex, random_real

pytestmark = pytest.mark.gpu

AXES = (-2, -1)
STOCKHAM = [impl for impl in spectral.FFT_IMPLS if impl.startswith("stockham")]
METHODS = [(impl, "transpose") for impl in spectral.FFT_IMPLS] + [
    (impl, "strided") for impl in STOCKHAM
]
METHOD_IDS = [f"{impl}-{cols}" for impl, cols in METHODS]

# (n1, n2): square and not, single rows and columns, the largest single-block columns, and shapes
# whose column count is not a multiple of a block's transforms.
SHAPES = [(1, 8), (8, 1), (2, 2), (16, 16), (64, 32), (8, 256), (256, 8), (4096, 4), (4, 4096)]
REAL_SHAPES = [(1, 8), (8, 2), (16, 16), (64, 32), (8, 256), (256, 8), (4096, 4), (2, 8192)]


def tolerance(n1: int, n2: int) -> float:
    return fft_tolerance(n1 * n2)


def cuda(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(x).cuda()


@pytest.mark.parametrize(("impl", "columns"), METHODS, ids=METHOD_IDS)
@pytest.mark.parametrize(("n1", "n2"), SHAPES)
def test_fft2_matches_the_reference(rng, impl, columns, n1, n2):
    x = random_complex(rng, (3, n1, n2))
    y = spectral.fft2(cuda(x), impl=impl, columns=columns)
    assert_transform_close(y, ref.fft2(x), n1 * n2, axes=AXES)


@pytest.mark.parametrize(("impl", "columns"), METHODS, ids=METHOD_IDS)
@pytest.mark.parametrize(("n1", "n2"), SHAPES)
def test_ifft2_matches_the_reference(rng, impl, columns, n1, n2):
    x = random_complex(rng, (3, n1, n2))
    y = spectral.fft2(cuda(x), inverse=True, impl=impl, columns=columns)
    assert_transform_close(y, ref.ifft2(x), n1 * n2, axes=AXES)


@pytest.mark.parametrize(("impl", "columns"), METHODS, ids=METHOD_IDS)
@pytest.mark.parametrize(("n1", "n2"), REAL_SHAPES)
def test_rfft2_matches_the_reference(rng, impl, columns, n1, n2):
    x = random_real(rng, (3, n1, n2))
    y = spectral.rfft2(cuda(x), impl=impl, columns=columns)
    assert y.shape == (3, n1, n2 // 2 + 1)
    assert_transform_close(y, ref.rfft2(x), n1 * n2, axes=AXES)


@pytest.mark.parametrize(("impl", "columns"), METHODS, ids=METHOD_IDS)
@pytest.mark.parametrize(("n1", "n2"), REAL_SHAPES)
def test_irfft2_matches_the_reference(rng, impl, columns, n1, n2):
    spectrum = ref.rfft2(random_real(rng, (3, n1, n2))).astype(np.complex64)
    y = spectral.irfft2(cuda(spectrum), n2, impl=impl, columns=columns)
    assert y.shape == (3, n1, n2)
    assert_transform_close(y, ref.irfft2(spectrum, (n1, n2)), n1 * n2, axes=AXES)


@pytest.mark.parametrize(("impl", "columns"), METHODS, ids=METHOD_IDS)
def test_irfft2_of_a_non_hermitian_spectrum_matches_torch(rng, impl, columns):
    spectrum = random_complex(rng, (2, 16, 9))  # n2 = 16
    expected = torch.fft.irfft2(torch.from_numpy(spectrum).to(torch.complex128), s=(16, 16))
    y = spectral.irfft2(cuda(spectrum), 16, impl=impl, columns=columns)
    assert_transform_close(y, expected.numpy(), 256, axes=AXES)


@pytest.mark.parametrize(("impl", "columns"), METHODS, ids=METHOD_IDS)
def test_round_trips(rng, impl, columns):
    x = cuda(random_complex(rng, (2, 32, 64)))
    back = spectral.fft2(
        spectral.fft2(x, impl=impl, columns=columns), inverse=True, impl=impl, columns=columns
    )
    assert_transform_close(back, x, 32 * 64, axes=AXES, tol=2 * tolerance(32, 64))

    r = cuda(random_real(rng, (2, 32, 64)))
    back = spectral.irfft2(spectral.rfft2(r, impl=impl, columns=columns), 64, impl=impl)
    assert_transform_close(back, r, 32 * 64, axes=AXES, tol=2 * tolerance(32, 64))


@pytest.mark.parametrize(("impl", "columns"), METHODS, ids=METHOD_IDS)
def test_many_arrays_and_leading_dimensions(rng, impl, columns):
    # 2 * 3 * 50 = 300 arrays of 8 x 16: the strided kernel's column blocks cross arrays.
    x = random_real(rng, (2, 3, 50, 8, 16))
    y = spectral.rfft2(cuda(x), impl=impl, columns=columns)
    assert y.shape == (2, 3, 50, 8, 9)
    assert_transform_close(y, ref.rfft2(x), 128, axes=AXES)


@pytest.mark.parametrize(("impl", "columns"), METHODS, ids=METHOD_IDS)
def test_non_contiguous_input(rng, impl, columns):
    x = cuda(random_complex(rng, (16, 32, 3))).permute(2, 0, 1)  # (3, 16, 32), strided
    assert not x.is_contiguous()
    y = spectral.fft2(x, impl=impl, columns=columns)
    assert_transform_close(y, ref.fft2(x.cpu().numpy()), 16 * 32, axes=AXES)


def test_auto_picks_strided_when_available(rng):
    # Same kernels, same order of operations: bit-identical results.
    x = cuda(random_real(rng, (4, 32, 32)))
    for impl in STOCKHAM:
        auto = spectral.rfft2(x, impl=impl)
        assert torch.equal(auto, spectral.rfft2(x, impl=impl, columns="strided"))
    naive = spectral.rfft2(x, impl="naive")
    assert torch.equal(naive, spectral.rfft2(x, impl="naive", columns="transpose"))


def test_empty_input_gives_empty_output():
    x = torch.zeros(0, 4, 8, device="cuda")
    assert spectral.rfft2(x).shape == (0, 4, 5)
    assert spectral.fft2(x.to(torch.complex64)).shape == (0, 4, 8)
    assert spectral.irfft2(torch.zeros(0, 4, 5, dtype=torch.complex64, device="cuda"), 8).shape == (
        0,
        4,
        8,
    )


# --- Rejected inputs --------------------------------------------------------------------------


def test_rejects_strided_without_a_column_kernel():
    x = torch.zeros(4, 8, dtype=torch.complex64, device="cuda")
    with pytest.raises(RuntimeError, match="no strided column kernel"):
        spectral.fft2(x, impl="naive", columns="strided")


def test_rejects_an_unknown_column_method():
    x = torch.zeros(4, 8, dtype=torch.complex64, device="cuda")
    with pytest.raises(RuntimeError, match="columns must be"):
        spectral.fft2(x, impl="naive", columns="sideways")


@pytest.mark.parametrize("impl", STOCKHAM)
def test_rejects_columns_longer_than_the_kernel_supports(impl):
    # Raised by the C++ driver as std::invalid_argument, which PyTorch turns into ValueError.
    x = torch.zeros(8192, 2, dtype=torch.complex64, device="cuda")
    with pytest.raises(ValueError, match="n1 to be a power of two up to 4096"):
        spectral.fft2(x, impl=impl)


def test_rejects_one_dimensional_input():
    with pytest.raises(RuntimeError, match="at least two dimensions"):
        spectral.rfft2(torch.zeros(8, device="cuda"))


def test_rejects_a_spectrum_of_the_wrong_width():
    x = torch.zeros(4, 8, dtype=torch.complex64, device="cuda")
    with pytest.raises(RuntimeError, match="n2/2 \\+ 1 = 5 input columns"):
        spectral.irfft2(x, 8)
