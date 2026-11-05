"""Radio signal processing on our FFTs (Task 1.9): window functions, a framed spectrum, and FIR
filtering by overlap-save, each with a selectable FFT backend.

    fft_impl="cufft"         torch.fft, which runs cuFFT: the library baseline
    fft_impl="stockham_r8"   one of our kernels (any name in spectral.FFT_IMPLS)

so every operation can be checked and timed with the library and with our own transforms, the
role `--fft-impl custom` plays in gpu-rf-pipeline. The same operations from the command line,
on the .cf32 / .f32 files of gpu-rf-pipeline (interleaved little-endian float32 I/Q, raw float32
taps):

    python -m spectral.rf spectrum --in x.cf32 --out spec.cf32 --nfft 1024 --window hann
    python -m spectral.rf fir --in x.cf32 --taps h.f32 --out y.cf32 --fft-impl stockham_r8

The framing around the transforms (zero padding, overlapping blocks, products with the filter's
spectrum) is plain PyTorch: the FFTs are the part being compared.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

from . import ops

FFT_BACKENDS: tuple[str, ...] = ("cufft", *ops.FFT_IMPLS)
WINDOWS: tuple[str, ...] = ("rect", "hann", "hamming", "blackman")


# --- FFT backend ------------------------------------------------------------------------------


def _check_backend(fft_impl: str) -> None:
    if fft_impl not in FFT_BACKENDS:
        raise ValueError(f"unknown fft_impl '{fft_impl}' (available: {', '.join(FFT_BACKENDS)})")


def fft(x: torch.Tensor, *, inverse: bool = False, fft_impl: str = "cufft") -> torch.Tensor:
    """Complex FFT along the last dimension (inverse divided by n) with the chosen backend."""
    _check_backend(fft_impl)
    if fft_impl == "cufft":
        return torch.fft.ifft(x) if inverse else torch.fft.fft(x)
    return ops.fft_c2c(x, inverse=inverse, impl=fft_impl)


# --- Windows ----------------------------------------------------------------------------------


def window(name: str, n: int, *, device: torch.device | str | None = None) -> torch.Tensor:
    """The periodic window of n points, as spectral analysis uses (scipy.signal.get_window with
    its default fftbins=True): computed in float64, returned as float32.

    rect      all ones
    hann      0.5 - 0.5 cos(2 pi k / n)
    hamming   0.54 - 0.46 cos(2 pi k / n)
    blackman  0.42 - 0.5 cos(2 pi k / n) + 0.08 cos(4 pi k / n)

    A window of one point is [1] whatever its name, as in scipy (the formulas would give 0).
    """
    if n < 1:
        raise ValueError(f"a window needs at least one point, got {n}")
    if name in WINDOWS and n == 1:
        return torch.ones(1, device=device)
    phase = 2.0 * np.pi * np.arange(n) / n
    if name == "rect":
        w = np.ones(n)
    elif name == "hann":
        w = 0.5 - 0.5 * np.cos(phase)
    elif name == "hamming":
        w = 0.54 - 0.46 * np.cos(phase)
    elif name == "blackman":
        w = 0.42 - 0.5 * np.cos(phase) + 0.08 * np.cos(2.0 * phase)
    else:
        raise ValueError(f"unknown window '{name}' (available: {', '.join(WINDOWS)})")
    return torch.from_numpy(w.astype(np.float32)).to(device)


# --- Spectrum ---------------------------------------------------------------------------------


def frames(x: torch.Tensor, nfft: int) -> torch.Tensor:
    """(..., n) -> (..., ceil(n / nfft), nfft): back-to-back frames, the last one zero-padded."""
    n = x.shape[-1]
    count = max(1, math.ceil(n / nfft))
    padded = torch.nn.functional.pad(x, (0, count * nfft - n))
    return padded.reshape(*x.shape[:-1], count, nfft)


def spectrum(
    x: torch.Tensor, nfft: int, *, window_name: str = "rect", fft_impl: str = "cufft"
) -> torch.Tensor:
    """FFT of each nfft-sample frame of x after applying the window: complex64 (..., n) ->
    (..., frames, nfft). With the rectangular window this is gpu-rf-pipeline's `rfgpu spectrum`.
    """
    _check_complex(x, "x")
    w = window(window_name, nfft, device=x.device)
    return fft(frames(x, nfft) * w, fft_impl=fft_impl)


# --- FIR by overlap-save ----------------------------------------------------------------------


def default_nfft(num_taps: int) -> int:
    """The block size fir() uses: a power of two about four times the filter length (so at
    least three quarters of each block is new output), from 64 up to 4096, the largest size our
    single-block kernels handle; longer filters get the next power of two above their length."""
    if num_taps < 1:
        raise ValueError("the filter needs at least one tap")
    nfft = 1 << max(6, math.ceil(math.log2(4 * num_taps)))
    if nfft > 4096:
        nfft = max(4096, 1 << math.ceil(math.log2(num_taps)))
    return nfft


def fir(
    x: torch.Tensor, taps: torch.Tensor, *, nfft: int | None = None, fft_impl: str = "cufft"
) -> torch.Tensor:
    """Causal FIR filter y[n] = sum_k taps[k] x[n - k] with zero initial state, as
    scipy.signal.lfilter(taps, 1, x): complex64 (..., n) -> (..., n), each row its own signal.

    Overlap-save: the signal is cut into blocks of nfft samples that overlap by M - 1 = len(taps)
    - 1, each block is transformed, multiplied by the spectrum of the zero-padded taps, and
    transformed back. Of each block's nfft circular-convolution outputs, the first M - 1 wrap
    around and are dropped; the remaining L = nfft - M + 1 are exact. taps may be real or
    complex.
    """
    _check_complex(x, "x")
    if taps.dim() != 1 or taps.numel() < 1:
        raise ValueError("taps must be a non-empty 1D tensor")
    m = taps.numel()
    nfft = default_nfft(m) if nfft is None else nfft
    if nfft < m or nfft & (nfft - 1):
        raise ValueError(f"nfft must be a power of two of at least len(taps) = {m}, got {nfft}")
    step = nfft - m + 1  # new outputs per block
    n = x.shape[-1]
    blocks = max(1, math.ceil(n / step))

    # Block b covers padded[b * step : b * step + nfft], where padded is x after m - 1 zeros.
    padded = torch.nn.functional.pad(x, (m - 1, blocks * step - n))
    segments = padded.unfold(-1, nfft, step).contiguous()  # (..., blocks, nfft)

    h = torch.zeros(nfft, dtype=torch.complex64, device=x.device)
    h[:m] = taps.to(device=x.device, dtype=torch.complex64)
    spectrum_h = fft(h, fft_impl=fft_impl)

    filtered = fft(fft(segments, fft_impl=fft_impl) * spectrum_h, inverse=True, fft_impl=fft_impl)
    y = filtered[..., m - 1 :].reshape(*x.shape[:-1], blocks * step)
    return y[..., :n].contiguous()


def _check_complex(x: torch.Tensor, name: str) -> None:
    if x.dtype != torch.complex64 or not x.is_cuda:
        raise ValueError(f"{name} must be a complex64 CUDA tensor, got {x.dtype} on {x.device}")


# --- Files and command line -------------------------------------------------------------------


def read_cf32(path: Path) -> np.ndarray:
    """Interleaved little-endian float32 I/Q samples as complex64."""
    return np.fromfile(path, dtype="<f4").view(np.complex64)


def write_cf32(path: Path, x: np.ndarray) -> None:
    np.asarray(x, dtype=np.complex64).view("<f4").tofile(path)


def read_f32(path: Path) -> np.ndarray:
    return np.fromfile(path, dtype="<f4")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m spectral.rf",
        description="Spectrum and FIR filtering of .cf32 recordings on the GPU.",
    )
    sub = parser.add_subparsers(dest="op", required=True)

    sp = sub.add_parser("spectrum", help="windowed FFT of back-to-back frames")
    sp.add_argument("--in", dest="inp", type=Path, required=True)
    sp.add_argument("--out", type=Path, required=True)
    sp.add_argument("--nfft", type=int, default=1024)
    sp.add_argument("--window", choices=WINDOWS, default="rect")

    fp = sub.add_parser("fir", help="FIR filter by overlap-save")
    fp.add_argument("--in", dest="inp", type=Path, required=True)
    fp.add_argument("--taps", type=Path, required=True, help="raw float32 taps (.f32)")
    fp.add_argument("--out", type=Path, required=True)
    fp.add_argument("--nfft", type=int, default=None, help="block size (default: about 4x taps)")

    for p in (sp, fp):
        p.add_argument("--fft-impl", choices=FFT_BACKENDS, default="cufft")
    args = parser.parse_args(argv)

    x = torch.from_numpy(read_cf32(args.inp)).cuda()
    if args.op == "spectrum":
        y = spectrum(x, args.nfft, window_name=args.window, fft_impl=args.fft_impl)
    else:
        taps = torch.from_numpy(read_f32(args.taps)).cuda()
        y = fir(x, taps, nfft=args.nfft, fft_impl=args.fft_impl)
    write_cf32(args.out, y.reshape(-1).cpu().numpy())
    print(f"{args.op}[{args.fft_impl}]: {x.numel()} samples in, {y.numel()} out -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
