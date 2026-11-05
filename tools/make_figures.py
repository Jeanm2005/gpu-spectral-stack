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

# Our fastest FFT implementation. Only it, cuFFT, and the scale kernel get a hue: categorical
# slots 1-3 of the reference palette, the only three that stay distinguishable in every pair,
# including under color-vision deficiency (checked with the palette validator in both modes).
# Our other implementations are drawn in gray, with their own marker shape and a direct label.
# (stockham_r8_table is newer than stockham_r8 but slower, so it stays gray.)
NEWEST = "stockham_r8"

# Every implementation, in drawing order; marker shapes are the second cue besides color.
ORDER = [
    "naive",
    "stockham_r2",
    "stockham_r4",
    "stockham_r8_table",
    "stockham_r8",
    "cufft",
    "ours",
]
MARKERS = {
    "cufft": "o",
    "naive": "^",
    "stockham_r2": "v",
    "stockham_r4": "s",
    "stockham_r8": "P",
    "stockham_r8_table": "X",
    "ours": "D",
}
LABELS = {
    "cufft": "cuFFT",
    "naive": "naive radix-2",
    "stockham_r2": "Stockham radix-2",
    "stockham_r8": "Stockham radix-8",
    "stockham_r8_table": "Stockham radix-8, twiddle table",
    "stockham_r4": "Stockham radix-4",
    "ours": "scale kernel (bandwidth reference)",
}

THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "text": "#0b0b0b",
        "muted": "#52514e",
        "grid": "#e6e5e1",
        "superseded": "#a9a8a2",
        "hues": {"cufft": "#2a78d6", "newest": "#eb6834", "ours": "#1baf7a"},
    },
    "dark": {
        "surface": "#1a1a19",
        "text": "#f0efec",
        "muted": "#c3c2b7",
        "grid": "#34332f",
        "superseded": "#6b6a64",
        "hues": {"cufft": "#3987e5", "newest": "#d95926", "ours": "#199e70"},
    },
}


def color_for(impl: str, theme: dict) -> str:
    if impl == NEWEST:
        return theme["hues"]["newest"]
    return theme["hues"].get(impl, theme["superseded"])


def is_highlighted(impl: str) -> bool:
    return impl in (NEWEST, "cufft", "ours")


def label_for(impl: str) -> str:
    return LABELS[impl] + (" (ours, newest)" if impl == NEWEST else "")


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


def present_impls(rows: list[dict]) -> list[str]:
    """The implementations in `rows`, in drawing order (gray ones first, under the others)."""
    found = {r["impl"] for r in rows}
    unknown = found - set(ORDER)
    if unknown:
        raise ValueError(f"add {sorted(unknown)} to ORDER, MARKERS, and LABELS in make_figures.py")
    return [impl for impl in ORDER if impl in found]


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


def finish(fig, ax, theme: dict, title: str, path: Path, **legend) -> None:
    ax.set_title(title, color=theme["text"], loc="left", fontsize=11)
    ax.legend(frameon=False, labelcolor=theme["text"], fontsize=8, **legend)
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
    ax.text(  # in the empty region under the sloped line, right of the FFT points
        ridge / 3,
        ridge / 3 * bw / 12,
        f"memory bandwidth\n{bw:,.0f} GB/s",
        color=theme["muted"],
        fontsize=9,
        ha="center",
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

    for impl in present_impls(rows):
        pts = [r for r in rows if r["impl"] == impl]
        ax.scatter(
            [r["gflops"] / r["gbps"] for r in pts],
            [r["gflops"] for r in pts],
            s=56 if is_highlighted(impl) else 40,
            marker=MARKERS[impl],
            color=color_for(impl, theme),
            edgecolors=theme["surface"],
            linewidths=1.5,
            label=label_for(impl),
            zorder=4 if is_highlighted(impl) else 3,
        )

    # Selective direct labels: only the smallest and largest cuFFT sizes.
    ffts = sorted(
        (r for r in rows if r["op"] == "fft_c2c" and r["impl"] == "cufft"), key=lambda r: r["n"]
    )
    for r in ffts[:1] + ffts[-1:]:
        ax.annotate(
            f"N = {r['n']}",
            (r["gflops"] / r["gbps"], r["gflops"]),
            xytext=(8, 6),
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
    finish(fig, ax, theme, f"Roofline, {peaks['device']}", path, loc="lower right")


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
    for impl in present_impls(ffts):
        pts = sorted((r for r in ffts if r["impl"] == impl), key=lambda r: r["n"])
        highlighted = is_highlighted(impl)
        ax.plot(
            [r["n"] for r in pts],
            [r["gbps"] for r in pts],
            color=color_for(impl, theme),
            linewidth=2 if highlighted else 1.5,
            marker=MARKERS[impl],
            markersize=7 if highlighted else 6,
            markeredgecolor=theme["surface"],
            markeredgewidth=1.5,
            label=label_for(impl),
            zorder=4 if highlighted else 3,
        )
        if not highlighted:  # gray lines carry a direct label, so identity is never color alone
            last = pts[-1]
            ax.annotate(
                LABELS[impl],
                (last["n"], last["gbps"]),
                xytext=(8, 0),
                textcoords="offset points",
                color=theme["muted"],
                fontsize=8,
                va="center",
            )

    ax.set_xscale("log", base=2)
    ax.set_xticks(sizes, [str(n) for n in sizes])
    ax.minorticks_off()
    ax.set_xlim(sizes[0] / 1.3, sizes[-1] * 4)  # room for the direct labels on the right
    ax.set_ylim(0, bw * 1.15)
    ax.set_xlabel("transform size N (points)")
    ax.set_ylabel("effective bandwidth (GB/s)")
    finish(
        fig,
        ax,
        theme,
        "Complex FFT effective bandwidth, 2^24 points per launch",
        path,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),  # outside the axes: the lines fill the whole plot
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
