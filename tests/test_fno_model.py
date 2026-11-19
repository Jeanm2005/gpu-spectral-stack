"""The FNO model (Tasks 3.2-3.3): the spectral convolution against the float64 NumPy reference
from Phase 0, the band of modes it keeps, and the full model's shapes and gradients. CPU only."""

import numpy as np
import pytest
import torch

from spectral import reference as ref
from spectral.fno.model import FNO2d, SpectralConv2d, relative_l2, spectral_conv2d_torch


def test_spectral_conv_matches_the_numpy_reference(rng):
    # Truncate with spectral.reference, mix each mode by its C_in x C_out matrix, zero-fill,
    # invert: the definition of the layer, in float64, independent of torch.fft.
    b, c_in, c_out, n1, n2, m1, m2 = 2, 3, 4, 16, 12, 3, 5
    x = rng.standard_normal((b, c_in, n1, n2))
    w = rng.standard_normal((c_in, c_out, 2 * m1, m2)) + 1j * rng.standard_normal(
        (c_in, c_out, 2 * m1, m2)
    )
    modes = ref.rfft2_truncate(x, m1, m2)  # (b, c_in, 2 m1, m2)
    mixed = np.einsum("bixy,ioxy->boxy", modes, w)
    expected = ref.irfft2_from_modes(mixed, (n1, n2))

    actual = spectral_conv2d_torch(torch.from_numpy(x), torch.from_numpy(w))
    np.testing.assert_allclose(actual.numpy(), expected, rtol=0, atol=1e-10)


def test_modes_outside_the_kept_band_are_removed():
    n = 32
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    low = np.cos(2 * np.pi * (1 * i + 2 * j) / n)
    high = np.cos(2 * np.pi * (10 * i + 12 * j) / n)
    x = torch.from_numpy(np.stack([low, high])[:, None])  # (2, 1, n, n)
    w = torch.ones(1, 1, 8, 4, dtype=torch.complex128)  # identity on the kept 8 x 4 modes
    y = spectral_conv2d_torch(x, w).numpy()[:, 0]
    np.testing.assert_allclose(y[0], low, rtol=0, atol=1e-12)  # kept unchanged
    np.testing.assert_allclose(y[1], 0, rtol=0, atol=1e-12)  # removed


def test_layer_rejects_more_modes_than_the_grid_has():
    layer = SpectralConv2d(2, 2, m1=9, m2=4)
    with pytest.raises(ValueError, match="do not fit"):
        layer(torch.zeros(1, 2, 16, 16))


def test_layer_rejects_an_unknown_backend():
    with pytest.raises(ValueError, match="unknown backend"):
        SpectralConv2d(2, 2, 4, 4, backend="nope")


def test_fno_shapes_and_gradients():
    torch.manual_seed(0)
    model = FNO2d(3, 2, modes=4, width=8, layers=2, projection=16)
    frames = torch.randn(5, 3, 16, 16)
    out = model(frames, torch.randn(5))
    assert out.shape == (5, 2, 16, 16)
    out.square().mean().backward()
    for name, p in model.named_parameters():
        grad = torch.view_as_real(p.grad) if p.grad.is_complex() else p.grad
        assert torch.isfinite(grad).all() and grad.abs().sum() > 0, name


def test_relative_l2():
    target = torch.randn(3, 1, 8, 8)
    torch.testing.assert_close(relative_l2(2 * target, target), torch.ones(3))
    torch.testing.assert_close(relative_l2(target, target), torch.zeros(3))
