"""Timing GPU work from Python, and reading and comparing benchmark results.

time_cuda() follows the C++ harness (csrc/common/bench.hpp): warm-up launches, then `rounds`
measurements of `iters` back-to-back launches between two CUDA events on the current stream,
reporting the median time per launch. Results use the same CSV columns as spectral_bench, so
the two can be compared row by row.
"""

from __future__ import annotations

import csv
import math
import statistics
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import torch

CSV_COLUMNS = ["op", "impl", "n", "batch", "time_ms", "min_ms", "max_ms", "gflops", "gbps"]

Key = tuple[str, str, int, int]  # (op, impl, n, batch)


@dataclass(frozen=True)
class Timing:
    median_ms: float  # all three are per launch
    min_ms: float
    max_ms: float


@dataclass(frozen=True)
class Result:
    op: str
    impl: str
    n: int
    batch: int
    timing: Timing
    flops: float
    nbytes: float


def time_cuda(fn: Callable[[], object], *, warmup: int = 3, rounds: int = 5, iters: int = 20):
    """Median, minimum, and maximum time per call of `fn`, which must enqueue GPU work on the
    current stream (any PyTorch CUDA operation does)."""
    if warmup < 0 or rounds < 1 or iters < 1:
        raise ValueError("need warmup >= 0, rounds >= 1, iters >= 1")
    for _ in range(warmup):
        fn()
    stream = torch.cuda.current_stream()
    per_launch = []
    for _ in range(rounds):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record(stream)
        for _ in range(iters):
            fn()
        stop.record(stream)
        stop.synchronize()
        per_launch.append(start.elapsed_time(stop) / iters)
    return Timing(statistics.median(per_launch), min(per_launch), max(per_launch))


def fft_c2c_work(n: int, batch: int) -> tuple[float, float]:
    """(flops, bytes) of `batch` complex transforms of n points, by the convention in
    csrc/common/bench.hpp: 5 n log2 n flops per transform, 16 bytes moved per point."""
    points = n * batch
    return 5.0 * points * math.log2(n), 16.0 * points


def fft_real_work(n: int, batch: int) -> tuple[float, float]:
    """(flops, bytes) of `batch` real transforms of n points, either direction, by the same
    convention: 2.5 n log2 n flops, n float32 values plus n/2 + 1 complex64 values moved."""
    return 2.5 * n * batch * math.log2(n), batch * (4.0 * n + 8.0 * (n / 2 + 1))


def format_csv(results: Iterable[Result], comments: Iterable[str]) -> str:
    lines = [f"# {c}" for c in comments]
    lines.append(",".join(CSV_COLUMNS))
    for r in results:
        t = r.timing
        lines.append(
            f"{r.op},{r.impl},{r.n},{r.batch},{t.median_ms:.6f},{t.min_ms:.6f},{t.max_ms:.6f},"
            f"{r.flops / (t.median_ms * 1e6):.1f},{r.nbytes / (t.median_ms * 1e6):.1f}"
        )
    return "\n".join(lines) + "\n"


def read_results(path: Path) -> dict[Key, float]:
    """Median time in ms for each (op, impl, n, batch) in a results CSV. Skips '#' lines."""
    with Path(path).open(newline="") as f:
        lines = [line for line in f if not line.startswith("#")]
    return {
        (row["op"], row["impl"], int(row["n"]), int(row["batch"])): float(row["time_ms"])
        for row in csv.DictReader(lines)
    }


def compare(
    old: dict[Key, float], new: dict[Key, float], threshold: float
) -> tuple[list[str], bool]:
    """A report line for each row, and whether any matched row is slower than
    old * (1 + threshold). Rows found in only one of the two are listed but never fail."""
    lines = [f"{'op':<10} {'impl':<12} {'n':>7} {'batch':>9} {'old ms':>10} {'new ms':>10} ratio"]
    regressed = False
    for key in sorted(old.keys() & new.keys()):
        ratio = new[key] / old[key]
        flag = ""
        if ratio > 1 + threshold:
            flag, regressed = "  SLOWER", True
        elif ratio < 1 - threshold:
            flag = "  faster"
        op, impl, n, batch = key
        lines.append(
            f"{op:<10} {impl:<12} {n:>7} {batch:>9} {old[key]:>10.4f} {new[key]:>10.4f} "
            f"{ratio:5.2f}x{flag}"
        )
    lines += [f"only in old: {key}" for key in sorted(old.keys() - new.keys())]
    lines += [f"only in new: {key}" for key in sorted(new.keys() - old.keys())]
    return lines, regressed
