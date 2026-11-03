"""Times torch.fft.fft (cuFFT behind PyTorch's dispatcher) with the sizes, batch rule, and CSV
columns of spectral_bench, so the cost PyTorch adds on top of a raw cuFFT call is visible.

    python bench/bench_torch.py --out docs/results/bench_torch.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from spectral.benchmark import Result, fft_c2c_work, format_csv, time_cuda


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--sizes", default=",".join(str(2**p) for p in range(1, 13)))
    parser.add_argument("--total", type=int, default=1 << 24, help="points per row")
    parser.add_argument("--out", type=Path, help="CSV file (default: standard output)")
    args = parser.parse_args(argv)
    if not torch.cuda.is_available():
        parser.error("needs a CUDA GPU")

    sizes = [int(s) for s in args.sizes.split(",")]
    data = torch.randn(max(args.total, max(sizes)), dtype=torch.complex64, device="cuda")
    results = []
    for n in sizes:
        batch = max(1, args.total // n)
        x = data[: n * batch].view(batch, n)
        timing = time_cuda(lambda x=x: torch.fft.fft(x))
        flops, nbytes = fft_c2c_work(n, batch)
        results.append(Result("fft_c2c", "torch", n, batch, timing, flops, nbytes))
        message = f"fft_c2c  torch  n={n:<6} batch={batch:<9} {timing.median_ms:9.4f} ms"
        print(message, file=sys.stderr)

    comments = [
        f"device: {torch.cuda.get_device_name()}",
        f"torch {torch.__version__}, CUDA {torch.version.cuda}",
        "timing: 3 warm-up launches, then the median over 5 rounds of 20 launches; "
        "time_ms is per launch",
    ]
    text = format_csv(results, comments)
    if args.out:
        args.out.write_text(text)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
