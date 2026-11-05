"""Times the radio operations of spectral.rf with each FFT backend: the same overlap-save FIR and
windowed spectrum around cuFFT and around each of our kernels, so the FFT is the only difference.

    python bench/bench_rf.py --out docs/results/bench_rf.csv

Rows use spectral_bench's CSV columns. op is fir_<taps> or spectrum_<nfft>, n the number of
samples, batch 1. The flops and bytes columns count only what the operation must do at minimum
(each sample read and written once; for the FIR, 8 flops per tap per output of a direct-form
complex-by-real filter), so GFLOP/s compare the backends, not the algorithms.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from spectral import rf
from spectral.benchmark import Result, format_csv, time_cuda


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--samples", type=int, default=1 << 22)
    parser.add_argument("--taps", default="63,1025", help="filter lengths")
    parser.add_argument("--nfft", default="1024", help="spectrum frame sizes")
    parser.add_argument("--out", type=Path, help="CSV file (default: standard output)")
    args = parser.parse_args(argv)
    if not torch.cuda.is_available():
        parser.error("needs a CUDA GPU")

    g = torch.Generator(device="cuda").manual_seed(0)
    n = args.samples
    x = torch.randn(n, dtype=torch.complex64, device="cuda", generator=g)
    results = []

    def record(op: str, impl: str, fn, flops: float) -> None:
        timing = time_cuda(fn)
        results.append(Result(op, impl, n, 1, timing, flops, 16.0 * n))
        print(f"{op:<14} {impl:<18} {timing.median_ms:9.4f} ms", file=sys.stderr)

    for m in (int(t) for t in args.taps.split(",")):
        taps = torch.randn(m, device="cuda", generator=g)
        for impl in rf.FFT_BACKENDS:
            record(f"fir_{m}", impl, lambda t=taps, i=impl: rf.fir(x, t, fft_impl=i), 8.0 * m * n)
    for nfft in (int(s) for s in args.nfft.split(",")):
        for impl in rf.FFT_BACKENDS:
            record(
                f"spectrum_{nfft}",
                impl,
                lambda f=nfft, i=impl: rf.spectrum(x, f, window_name="hann", fft_impl=i),
                5.0 * n * (nfft.bit_length() - 1),
            )

    comments = [
        f"device: {torch.cuda.get_device_name()}",
        f"torch {torch.__version__}, CUDA {torch.version.cuda}",
        "timing: 3 warm-up calls, then the median over 5 rounds of 20 calls; time_ms is per call",
    ]
    text = format_csv(results, comments)
    if args.out:
        args.out.write_text(text)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
