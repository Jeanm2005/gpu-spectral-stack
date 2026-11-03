"""Draws the figures in docs/figures/ from the committed results in docs/results/.

    python tools/make_figures.py

Each figure is written in a light and a dark version (name.png and name-dark.png), for the
README's light and dark themes:

    roofline        every benchmark row as a point (arithmetic intensity, GFLOP/s) under the two
                    measured ceilings, memory bandwidth and FP32 throughput
    fft_bandwidth   effective bandwidth of each FFT implementation by transform size, against
                    the measured copy bandwidth

bench_torch.csv is not plotted: torch.fft runs cuFFT, and its curve would sit on cuFFT's.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

# Colors follow the implementation, never its position in a list, so each keeps its color in
# every figure. These are categorical slots 1-3 of the reference palette, the only three that
# stay distinguishable in every pair, including under color-vision deficiency (checked with the
# palette validator in both modes). Distinct marker shapes are the second cue. When more
# implementations arrive, superseded ones become gray rather than taking a fourth hue.
THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "text": "#0b0b0b",
        "muted": "#52514e",
        "grid": "#e6e5e1",
        "series": {"cufft": "#2a78d6", "naive": "#eb6834", "ours": "#1baf7a"},
    },
    "dark": {
        "surface": "#1a1a19",
        "text": "#f0efec",
        "muted": "#c3c2b7",
        "grid": "#34332f",
        "series": {"cufft": "#3987e5", "naive": "#d95926", "ours": "#199e70"},
    },
}
MARKERS = {"cufft": "o", "naive": "^", "ours": "D"}
LABELS = {
    "cufft": "cuFFT",
    "naive": "naive radix-2 (ours)",
    "ours": "scale kernel (bandwidth reference)",
}


def load_rows(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        lines = [line for line in f if not line.startswith("#")]
    return [
        {
            "op": row["op"],
            "impl": row["impl"],
            "n": int(row["n"]),
            "batch": int(row["batch"]),
            "gflops": float(row["gflops"]),
            "gbps": float(row["gbps"]),
        }
        for row in csv.DictReader(lines)
    ]


def present_impls(rows: list[dict], theme: dict) -> list[str]:
    """The implementations in `rows`, in the fixed color order."""
    found = {r["impl"] for r in rows}
    unknown = found - theme["series"].keys()
    if unknown:
        raise ValueError(f"add a color for {sorted(unknown)} to THEMES in make_figures.py")
    return [impl for impl in theme["series"] if impl in found]


def styled_axes(theme: dict):
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    fig.patch.set_facecolor(theme["surface"])
    ax.set_facecolor(theme["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme["muted"])
    ax.tick_params(colors=theme["muted"], which="both")
    ax.xaxis.label.set_color(theme["muted"])
    ax.yaxis.label.set_color(theme["muted"])
    ax.grid(True, which="major", color=theme["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    return fig, ax


def finish(fig, ax, theme: dict, title: str, path: Path, legend_loc: str) -> None:
    ax.set_title(title, color=theme["text"], loc="left", fontsize=11)
    ax.legend(frameon=False, labelcolor=theme["text"], loc=legend_loc)
    fig.savefig(path, facecolor=theme["surface"], bbox_inches="tight")
    plt.close(fig)


def roofline(rows: list[dict], peaks: dict, theme: dict, path: Path) -> None:
    bw = peaks["peak_bandwidth_gbps"]
    fp32 = peaks["peak_fp32_gflops"]
    ridge = fp32 / bw
    intensity = [r["gflops"] / r["gbps"] for r in rows]  # flops / bytes of each row
    lo = min(min(intensity) / 2, 0.05)
    hi = max(max(intensity) * 2, ridge * 4)
    xs = np.geomspace(lo, hi, 256)
    bottom = min(min(r["gflops"] for r in rows) / 2, lo * bw / 2)

    fig, ax = styled_axes(theme)
    ax.plot(xs, np.minimum(fp32, xs * bw), color=theme["muted"], linewidth=2, zorder=2)
    ax.axvline(ridge, color=theme["muted"], linewidth=1, linestyle=":", zorder=1)
    ax.text(
        lo * 1.3,
        lo * 1.3 * bw * 2,
        f"memory bandwidth {bw:,.0f} GB/s",
        color=theme["muted"],
        fontsize=9,
    )
    ax.text(
        hi / 1.2,
        fp32 * 1.3,
        f"FP32 {fp32 / 1000:,.1f} TFLOP/s",
        color=theme["muted"],
        fontsize=9,
        ha="right",
    )
    ax.text(
        ridge * 1.15,
        np.sqrt(bottom * fp32),
        f"ridge point\n{ridge:.0f} flop/byte",
        color=theme["muted"],
        fontsize=9,
    )

    for impl in present_impls(rows, theme):
        pts = [r for r in rows if r["impl"] == impl]
        ax.scatter(
            [r["gflops"] / r["gbps"] for r in pts],
            [r["gflops"] for r in pts],
            s=56,
            marker=MARKERS[impl],
            color=theme["series"][impl],
            edgecolors=theme["surface"],
            linewidths=1.5,
            label=LABELS[impl],
            zorder=3,
        )

    # Selective direct labels: only the smallest and largest cuFFT sizes.
    ffts = sorted(
        (r for r in rows if r["op"] == "fft_c2c" and r["impl"] == "cufft"), key=lambda r: r["n"]
    )
    for r in ffts[:1] + ffts[-1:]:
        ax.annotate(
            f"N = {r['n']}",
            (r["gflops"] / r["gbps"], r["gflops"]),
            xytext=(8, -14),
            textcoords="offset points",
            color=theme["text"],
            fontsize=8,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(bottom, fp32 * 3)
    ax.set_xlabel("arithmetic intensity (flop/byte)")
    ax.set_ylabel("GFLOP/s")
    finish(fig, ax, theme, f"Roofline, {peaks['device']}", path, "lower right")


def fft_bandwidth(rows: list[dict], peaks: dict, theme: dict, path: Path) -> None:
    ffts = [r for r in rows if r["op"] == "fft_c2c"]
    bw = peaks["peak_bandwidth_gbps"]
    sizes = sorted({r["n"] for r in ffts})

    fig, ax = styled_axes(theme)
    ax.axhline(bw, color=theme["muted"], linewidth=1, linestyle="--", zorder=1)
    ax.text(
        sizes[0],
        bw * 1.02,
        "measured copy bandwidth",
        color=theme["muted"],
        fontsize=9,
        va="bottom",
    )
    for impl in present_impls(ffts, theme):
        pts = sorted((r for r in ffts if r["impl"] == impl), key=lambda r: r["n"])
        ax.plot(
            [r["n"] for r in pts],
            [r["gbps"] for r in pts],
            color=theme["series"][impl],
            linewidth=2,
            marker=MARKERS[impl],
            markersize=7,
            markeredgecolor=theme["surface"],
            markeredgewidth=1.5,
            label=LABELS[impl],
            zorder=3,
        )

    ax.set_xscale("log", base=2)
    ax.set_xticks(sizes, [str(n) for n in sizes])
    ax.minorticks_off()
    ax.set_ylim(0, bw * 1.15)
    ax.set_xlabel("transform size N (points)")
    ax.set_ylabel("effective bandwidth (GB/s)")
    finish(
        fig,
        ax,
        theme,
        "Complex FFT effective bandwidth, 2^24 points per launch",
        path,
        "upper right",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--results", type=Path, default=ROOT / "docs" / "results")
    parser.add_argument("--figures", type=Path, default=ROOT / "docs" / "figures")
    args = parser.parse_args(argv)

    rows = load_rows(args.results / "bench.csv")
    peaks = json.loads((args.results / "peaks.json").read_text())

    args.figures.mkdir(parents=True, exist_ok=True)
    for name, theme in THEMES.items():
        suffix = "" if name == "light" else f"-{name}"
        roofline(rows, peaks, theme, args.figures / f"roofline{suffix}.png")
        fft_bandwidth(rows, peaks, theme, args.figures / f"fft_bandwidth{suffix}.png")
    print(f"wrote {2 * len(THEMES)} figures to {args.figures}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
