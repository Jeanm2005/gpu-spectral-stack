"""How well the loader keeps the GPU busy (Task 2.7): samples per second and GPU idle time,
measured with the PyTorch profiler, for several loader settings.

    python tools/loader_profile.py data/flow.zarr --out docs/results/loader.csv

The consumer stands in for a training step: on each batch it runs a fixed amount of GPU work,
`--work` round trips of our rfft2 / irfft2 over the input frames, and nothing else. It is not a
model: Phase 3 brings that. Each setting runs at several amounts of work: 0 measures what the
loader alone delivers, and the larger ones whether the GPU waits for data once it has real
work to do.

GPU busy and idle time come from the profiler's trace, which lists every kernel and copy with
its start and length: busy = time covered by at least one of them, idle = 1 - busy / (first
start to last end), over the profiled steps. Busy time per step is in absolute milliseconds,
so it shows when the GPU itself ran slower (a clock that is not locked) rather than idled.

Measure on the charger with the clock locked (docs/DEVELOPMENT.md); on battery the GPU clock
collapses and every number here changes.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

import spectral
from spectral.flow.loader import CudaPrefetcher, WindowDataset, make_loader

SETTINGS = [  # (workers, pin_memory, prefetcher)
    (0, False, False),
    (2, True, True),
    (4, False, False),
    (4, True, False),
    (4, True, True),
    (8, True, True),
]


def gpu_busy(trace_path: Path) -> tuple[float, float]:
    """(busy, span) in microseconds: time covered by at least one kernel or copy, and the time
    from the first GPU activity to the last."""
    events = json.loads(trace_path.read_text())["traceEvents"]
    spans = sorted(
        (e["ts"], e["ts"] + e["dur"])
        for e in events
        if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and "dur" in e
    )
    if not spans:
        return 0.0, 0.0
    busy, cur_start, cur_end = 0.0, *spans[0]
    for start, end in spans[1:]:
        if start > cur_end:
            busy += cur_end - cur_start
            cur_start, cur_end = start, end
        else:
            cur_end = max(cur_end, end)
    busy += cur_end - cur_start
    # The last activity to end need not be the last to start (a copy inside a long kernel).
    return busy, max(end for _, end in spans) - spans[0][0]


def consume(batch: dict, work: int) -> None:
    x = batch["input"]
    if not x.is_cuda:
        x = x.cuda(non_blocking=True)
    for _ in range(work):
        x = spectral.irfft2(spectral.rfft2(x, impl="stockham_r8"), x.shape[-1], impl="stockham_r8")


def drop_cache(path: Path) -> None:
    """Evicts the dataset's files from the page cache (no root needed), so every setting starts
    reading from disk rather than from what the previous setting left in memory."""
    os.sync()
    for f in Path(path).rglob("*"):
        if f.is_file():
            fd = os.open(f, os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)


def run(path, workers, pin, prefetch, batch_size, steps, warmup, work, inputs):
    drop_cache(path)
    ds = WindowDataset(path, inputs=inputs, outputs=1, seed=0)
    loader = make_loader(ds, batch_size, num_workers=workers, pin_memory=pin)
    source = CudaPrefetcher(loader) if prefetch else loader
    it = iter(source)
    for _ in range(warmup):  # workers started, first batches in flight
        consume(next(it), work)
    torch.cuda.synchronize()
    with tempfile.TemporaryDirectory() as tmp:
        trace = Path(tmp) / "trace.json"
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            start = time.perf_counter()
            for _ in range(steps):
                consume(next(it), work)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start
        prof.export_chrome_trace(str(trace))
        busy, span = gpu_busy(trace)
    del it, source, loader
    return {
        "work": work,
        "workers": workers,
        "pin_memory": pin,
        "prefetcher": prefetch,
        "samples_per_s": round(steps * batch_size / seconds, 1),
        "step_ms": round(1000 * seconds / steps, 2),
        "gpu_busy_ms_per_step": round(busy / 1000 / steps, 2),
        "gpu_idle_pct": round(100 * (1 - busy / span), 1) if span else 100.0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--inputs", type=int, default=10)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument(
        "--work",
        default="0,4,32",
        help="rfft2/irfft2 round trips per batch, one run of every setting for each: 0 measures "
        "the loader alone, larger values a consumer with that much GPU work",
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    rows = []
    for work in (int(w) for w in args.work.split(",")):
        for workers, pin, prefetch in SETTINGS:
            row = run(args.dataset, workers, pin, prefetch, args.batch_size, args.steps,
                      args.warmup, work, args.inputs)  # fmt: skip
            rows.append(row)
            print(row, file=sys.stderr)

    header = (
        f"# {args.dataset}: batches of {args.batch_size} windows ({args.inputs} frames in, 1 out); "
        f"work = rfft2/irfft2 round trips per batch; {args.steps} profiled steps after "
        f"{args.warmup} warm-up"
    )
    out = args.out.open("w", newline="") if args.out else sys.stdout
    try:
        out.write(header + "\n")
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    finally:
        if args.out:
            out.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
