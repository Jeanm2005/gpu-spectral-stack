"""Smoke tests for the PyTorch extension: the operator is registered, runs our kernel, launches
on PyTorch's current stream, and rejects inputs it does not support."""

import pytest
import torch

spectral = pytest.importorskip(
    "spectral", reason="extension not built: pip install -e . --no-build-isolation"
)
needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def test_operator_is_registered():
    assert hasattr(torch.ops.spectral, "scale")


# 255/256/257 straddle one block of 256 threads; 2^20 needs many blocks.
@needs_gpu
@pytest.mark.parametrize("n", [1, 255, 256, 257, 1 << 20])
def test_scale_matches_torch(n):
    g = torch.Generator(device="cuda").manual_seed(n)
    x = torch.randn(n, dtype=torch.complex64, device="cuda", generator=g)
    # Multiplying by 0.5 is exact in floating point, so the results must match bit for bit.
    torch.testing.assert_close(spectral.scale(x, 0.5), x * 0.5, rtol=0, atol=0)


@needs_gpu
def test_scale_non_contiguous_input():
    x = torch.randn(1000, dtype=torch.complex64, device="cuda")[::2]
    assert not x.is_contiguous()
    torch.testing.assert_close(spectral.scale(x, 0.5), x * 0.5, rtol=0, atol=0)


@needs_gpu
def test_scale_on_side_stream():
    x = torch.randn(1 << 20, dtype=torch.complex64, device="cuda")
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())  # x was produced on the default stream
    with torch.cuda.stream(side):
        y = spectral.scale(x, 0.5)
    side.synchronize()
    torch.testing.assert_close(y, x * 0.5, rtol=0, atol=0)


def test_cpu_input_is_rejected_by_the_dispatcher():
    x = torch.zeros(4, dtype=torch.complex64)
    with pytest.raises(NotImplementedError, match="CPU"):
        spectral.scale(x, 0.5)


@needs_gpu
def test_wrong_dtype_is_rejected():
    x = torch.zeros(4, dtype=torch.float32, device="cuda")
    with pytest.raises(RuntimeError, match="complex64"):
        spectral.scale(x, 0.5)