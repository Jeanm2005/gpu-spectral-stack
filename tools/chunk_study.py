"""Chunk-size study for the dataset layout (Task 2.4): write speed, size, and cold read speed of
each candidate layout, under the reads training will make.

    python tools/chunk_study.py --out docs/results/chunk_study.csv

One dataset (simulated once on the GPU) is written in every layout. Before each read pattern
the page cache is dropped for the dataset's files (posix_fadvise DONTNEED, no root needed), so
reads come from disk, as they will for a dataset larger than RAM. Read patterns:

    trajectory  whole trajectories of random samples (validation, evaluation)
    window      random windows of 11 consecutive frames (10 in, 1 out: an FNO training sample)
    frame       random single frames (one-step training pairs read as two of these)
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from spectral.flow.generate import DatasetConfig
from spectral.flow.store import create, write_dataset

WINDOW = 11


def drop_cache(path: Path) -> None:
    os.sync()
    for f in path.rglob("*"):
        if f.is_file():
            fd = os.open(f, os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)


def disk_usage(path: Path) -> tuple[int, int]:
    files = [f for f in path.rglob("*") if f.is_file()]
    return sum(f.stat().st_size for f in files), len(files)


def timed_reads(arr, pattern: str, rng: np.random.Generator) -> tuple[float, int, int]:
    """(seconds, items, bytes) for one read pattern."""
    s, t = arr.shape[:2]
    frame_bytes = arr.shape[2] * arr.shape[3] * 4
    start = time.perf_counter()
    if pattern == "trajectory":
        items = 16
        for i in rng.choice(s, items, replace=False):
            arr[int(i)]
        nbytes = items * t * frame_bytes
    elif pattern == "window":
        items = 256
        for _ in range(items):
            i, j = int(rng.integers(s)), int(rng.integers(t - WINDOW + 1))
            arr[i, j : j + WINDOW]
        nbytes = items * WINDOW * frame_bytes
    else:
        items = 512
        for _ in range(items):
            arr[int(rng.integers(s)), int(rng.integers(t))]
        nbytes = items * frame_bytes
    return time.perf_counter() - start, items, nbytes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--frames", type=int, default=41)
    parser.add_argument("--chunk-frames", default="1,8,41")
    parser.add_argument("--out", type=Path, help="CSV file (default: standard output)")
    parser.add_argument("--workdir", type=Path, default=None, help="where to write the layouts")
    args = parser.parse_args(argv)

    config = DatasetConfig(
        n=args.n,
        dt=2.5e-3,
        record_every=10,
        frames=args.frames,
        num_samples=args.samples,
        batch_size=args.samples,
        viscosity_min=1e-4,
        viscosity_max=1e-3,
    )
    work = Path(tempfile.mkdtemp(dir=args.workdir))
    try:
        source = write_dataset(work / "source.zarr", config)
        data = source.vorticity[:]
        rows = []
        for chunk_frames in (int(c) for c in args.chunk_frames.split(",")):
            for compression in (False, True):
                path = work / f"c{chunk_frames}_{'zstd' if compression else 'raw'}.zarr"
                ds = create(path, config, chunk_frames=chunk_frames, compression=compression)
                os.sync()
                start = time.perf_counter()
                ds.vorticity[:] = data
                os.sync()
                write_s = time.perf_counter() - start
                size, files = disk_usage(path / "vorticity")
                row = {
                    "chunk_frames": chunk_frames,
                    "compression": "zstd1" if compression else "none",
                    "files": files,
                    "size_mb": round(size / 1e6, 1),
                    "write_mb_s": round(data.nbytes / 1e6 / write_s, 1),
                }
                for pattern in ("trajectory", "window", "frame"):
                    drop_cache(path)
                    seconds, items, nbytes = timed_reads(
                        ds.vorticity, pattern, np.random.default_rng(0)
                    )
                    row[f"{pattern}_per_s"] = round(items / seconds, 1)
                    row[f"{pattern}_mb_s"] = round(nbytes / 1e6 / seconds, 1)
                rows.append(row)
                print(row, file=sys.stderr)
                shutil.rmtree(path)
    finally:
        shutil.rmtree(work, ignore_errors=True)

    header = [
        f"# n={args.n}, samples={args.samples}, frames={args.frames}, float32, "
        f"{args.samples * args.frames * args.n * args.n * 4 / 1e6:.0f} MB; "
        f"samples_per_shard=16; reads after dropping the page cache; window = {WINDOW} frames"
    ]
    out = args.out.open("w", newline="") if args.out else sys.stdout
    try:
        out.write(header[0] + "\n")
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    finally:
        if args.out:
            out.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
