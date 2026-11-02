#!/usr/bin/env python3
"""NumPy/SciPy CPU reference for every rfgpu operation.

This file is the definition of "correct": each GPU result is compared to the function of the
same name here. Everything is computed in float64, so the reference is more accurate than the
float32 GPU output and the test tolerances only have to cover the GPU's rounding.

Output conventions (the GPU kernels must match these exactly):

  spectrum  The input is cut into consecutive frames of `nfft` samples, no overlap and no
            window. A partial last frame is zero-padded, so there are ceil(N / nfft) frames.
            Each frame is a forward FFT, unnormalized (what cuFFT computes), with bin 0 first.
            Output: num_frames * nfft complex values, frame after frame.

  fir       y[n] = sum_{k=0}^{T-1} taps[k] * x[n - k], with x[m] = 0 for m < 0.
            Causal, zero initial state, real taps. Output length equals input length N, so
            the first T - 1 outputs are the start-up transient.

  xcorr     r[k] = sum_n b[n + k] * conj(a[n]) for every lag k from -(Na - 1) to Nb - 1,
            with samples outside either input taken as zero. Output length Na + Nb - 1, and
            output index i holds lag i - (Na - 1). If b is a delayed by d samples, |r| peaks
            at lag +d.

  channelize  Critically sampled polyphase filter bank with M channels and a prototype
            low-pass h (length L, taken as zero past its end):
              z[t][k] = sum_l h[l] * x[t*M - l] * exp(+2j*pi*k*l / M),  x[m] = 0 for m < 0
            for t = 0 ... ceil(N / M) - 1 and k = 0 ... M - 1. Channel k is the band around
            k * fs / M (k > M / 2 are the negative frequencies), decimated to fs / M. Output:
            frame after frame of M channels, like spectrum. Computed as M polyphase branch sums
            followed by an M-point inverse FFT, unnormalized.

  gcc_phat  Generalized cross-correlation with phase transform. With L the smallest power of
            two >= Na + Nb - 1 and A, B the L-point FFTs of the zero-padded inputs:
              W[f] = B[f] conj(A[f]) / |B[f] conj(A[f])|   (0 where the product is 0)
            and r = IFFT(W), normalized by 1/L, in the same lag order as xcorr. Every frequency
            gets unit weight, so the peak is sharp, and for b = a delayed by d it is 1 at lag d.

  doa       Direction of arrival for a uniform linear array: channels 0 ... K-1 at spacing d
            meters, propagation speed c, sample rate fs. tau_i = (argmax_k |gcc_phat(ch0,
            ch_i)[k]|) / fs is the delay of channel i behind channel 0; the least-squares fit of
            tau_i = i * d * sin(theta) / c gives theta in degrees, positive when the wave reaches
            channel 0 first.

Usage (mirrors the rfgpu CLI):
    python tools/reference.py spectrum --in tests/fixtures/tone.cf32 --out out/ref_spectrum.cf32 --nfft 1024
    python tools/reference.py fir      --in tests/fixtures/tone.cf32 --taps tests/fixtures/lowpass.f32 --out out/ref_fir.cf32
    python tools/reference.py xcorr    --a tests/fixtures/ch0.cf32 --b tests/fixtures/ch1.cf32 --out out/ref_xcorr.cf32
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy import signal


# --- File I/O ------------------------------------------------------------------------------

def read_cf32(path: Path) -> np.ndarray:
    """Read interleaved little-endian float32 [I0, Q0, I1, Q1, ...] as complex128."""
    iq = np.fromfile(path, dtype="<f4")
    if len(iq) % 2 != 0:
        raise ValueError(f"{path}: odd number of float32 values, not interleaved IQ")
    return iq[0::2].astype(np.float64) + 1j * iq[1::2].astype(np.float64)


def write_cf32(path: Path, x: np.ndarray) -> None:
    iq = np.empty(2 * x.size, dtype="<f4")
    iq[0::2] = x.real.ravel()
    iq[1::2] = x.imag.ravel()
    path.parent.mkdir(parents=True, exist_ok=True)
    iq.tofile(path)


def read_f32(path: Path) -> np.ndarray:
    """Read a raw little-endian float32 array (filter taps) as float64."""
    return np.fromfile(path, dtype="<f4").astype(np.float64)


def write_f32(path: Path, x: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x.ravel().astype("<f4").tofile(path)


# --- Operations ----------------------------------------------------------------------------

def spectrum(x: np.ndarray, nfft: int) -> np.ndarray:
    """Framed FFT. Returns shape (ceil(len(x) / nfft), nfft), complex128."""
    if nfft < 1:
        raise ValueError("nfft must be at least 1")
    num_frames = -(-len(x) // nfft)  # ceiling division
    padded = np.zeros(num_frames * nfft, dtype=np.complex128)
    padded[: len(x)] = x
    return np.fft.fft(padded.reshape(num_frames, nfft), axis=1)


def magnitude(frames: np.ndarray) -> np.ndarray:
    """|X| of each bin, float64, same shape as the input."""
    return np.abs(frames)


def fir(x: np.ndarray, taps: np.ndarray) -> np.ndarray:
    """Causal FIR filter with zero initial state. Returns len(x) complex128 samples."""
    if len(taps) < 1:
        raise ValueError("need at least one tap")
    if len(x) == 0:
        return np.zeros(0, dtype=np.complex128)
    # Full convolution has len(x) + len(taps) - 1 samples; the first len(x) are the causal
    # output. (np.convolve rather than scipy.signal.lfilter, which is slow for long filters.)
    return np.convolve(x.astype(np.complex128), taps.astype(np.float64))[: len(x)]


def xcorr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Full cross-correlation of b against a. Returns len(a) + len(b) - 1 complex128 values."""
    if len(a) == 0 or len(b) == 0:
        raise ValueError("both inputs need at least one sample")
    # scipy conjugates its second argument: correlate(b, a)[i] = sum_n b[n + k] * conj(a[n])
    # with k = i - (len(a) - 1).
    return signal.correlate(b.astype(np.complex128), a.astype(np.complex128), mode="full")


def xcorr_lags(num_a: int, num_b: int) -> np.ndarray:
    """Lag of each xcorr output index: -(num_a - 1) ... num_b - 1."""
    return np.arange(-(num_a - 1), num_b)


def channelize(x: np.ndarray, taps: np.ndarray, channels: int) -> np.ndarray:
    """Polyphase channelizer. Returns shape (ceil(len(x) / channels), channels), complex128."""
    m = channels
    if m < 1 or len(taps) < 1:
        raise ValueError("need at least one channel and one tap")
    frames = -(-len(x) // m)
    branches = -(-len(taps) // m)
    h = np.zeros(branches * m)
    h[: len(taps)] = taps
    # Padded so x[t*M - p*M - m] is a plain index for every t, p, m: index + pad.
    pad = branches * m
    xp = np.concatenate([np.zeros(pad, dtype=np.complex128), x.astype(np.complex128)])
    u = np.zeros((frames, m), dtype=np.complex128)
    t = np.arange(frames)
    for p in range(branches):
        for branch in range(m):
            u[:, branch] += h[p * m + branch] * xp[pad + t * m - p * m - branch]
    # sum_m u[m] exp(+2j pi k m / M) is M * ifft.
    return np.fft.ifft(u, axis=1) * m


def gcc_phat(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """GCC-PHAT of b against a, len(a) + len(b) - 1 complex128 values in xcorr's lag order."""
    if len(a) == 0 or len(b) == 0:
        raise ValueError("both inputs need at least one sample")
    num_lags = len(a) + len(b) - 1
    length = 1
    while length < num_lags:
        length *= 2
    cross = np.fft.fft(b, length) * np.conj(np.fft.fft(a, length))
    mag = np.abs(cross)
    weighted = np.where(mag > 0, cross / np.where(mag > 0, mag, 1), 0)
    circular = np.fft.ifft(weighted)
    # Lags -(Na - 1) ... -1 wrap to the end of the circular result.
    return np.concatenate([circular[length - (len(a) - 1):], circular[: len(b)]])


def tdoa(a: np.ndarray, b: np.ndarray) -> int:
    """Delay of b behind a in samples: the lag of the GCC-PHAT peak."""
    r = gcc_phat(a, b)
    return int(xcorr_lags(len(a), len(b))[np.argmax(np.abs(r))])


def doa(channels: list[np.ndarray], spacing: float, speed: float, rate: float) -> float:
    """Direction of arrival in degrees for a uniform linear array (see the module docstring)."""
    if len(channels) < 2:
        raise ValueError("need at least two channels")
    index = np.arange(1, len(channels))
    taus = np.array([tdoa(channels[0], ch) for ch in channels[1:]]) / rate
    slope = float(np.sum(index * taus) / np.sum(index * index))
    return float(np.degrees(np.arcsin(np.clip(speed * slope / spacing, -1.0, 1.0))))


# --- CLI -----------------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="op", required=True)

    p = sub.add_parser("spectrum", help="framed FFT")
    p.add_argument("--in", dest="inp", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--nfft", type=int, default=1024)

    p = sub.add_parser("fir", help="FIR filter")
    p.add_argument("--in", dest="inp", type=Path, required=True)
    p.add_argument("--taps", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)

    for name, text in (("xcorr", "cross-correlation"), ("gcc_phat", "GCC-PHAT")):
        p = sub.add_parser(name, help=text)
        p.add_argument("--a", type=Path, required=True)
        p.add_argument("--b", type=Path, required=True)
        p.add_argument("--out", type=Path, required=True)

    p = sub.add_parser("channelize", help="polyphase channelizer")
    p.add_argument("--in", dest="inp", type=Path, required=True)
    p.add_argument("--taps", type=Path, required=True)
    p.add_argument("--channels", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)

    args = parser.parse_args()
    if args.op == "spectrum":
        y = spectrum(read_cf32(args.inp), args.nfft)
    elif args.op == "fir":
        y = fir(read_cf32(args.inp), read_f32(args.taps))
    elif args.op == "channelize":
        y = channelize(read_cf32(args.inp), read_f32(args.taps), args.channels)
    elif args.op == "gcc_phat":
        y = gcc_phat(read_cf32(args.a), read_cf32(args.b))
    else:
        y = xcorr(read_cf32(args.a), read_cf32(args.b))
    write_cf32(args.out, y)
    print(f"{args.op}: wrote {y.size} complex samples to {args.out}")


if __name__ == "__main__":
    main()
