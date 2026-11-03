"""Compares two benchmark CSVs and fails if any result got slower than the threshold allows.

    python tools/bench_compare.py docs/results/bench.csv out/bench.csv
    python tools/bench_compare.py old.csv new.csv --threshold 0.05

Rows are matched by (op, impl, n, batch) and compared by their median time. Exit status 1 if any
matched row is slower than old * (1 + threshold); rows found in only one file are listed but do
not fail the comparison.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from spectral.benchmark import compare, read_results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("old", type=Path, help="baseline CSV (usually docs/results/...)")
    parser.add_argument("new", type=Path, help="CSV from the change being checked")
    parser.add_argument("--threshold", type=float, default=0.10, help="allowed slowdown (0.10)")
    args = parser.parse_args(argv)

    lines, regressed = compare(read_results(args.old), read_results(args.new), args.threshold)
    print("\n".join(lines))
    if regressed:
        print(f"\nRegression: at least one row is more than {args.threshold:.0%} slower.")
    return 1 if regressed else 0


if __name__ == "__main__":
    sys.exit(main())
