"""Generates the radio test signals in tests/fixtures/rf/ (Task 1.9). Deterministic: fixed seeds.

    python tools/gen_rf_fixtures.py                    # rewrite tests/fixtures/rf/
    python tools/gen_rf_fixtures.py --out /tmp/check   # anywhere else (the tests compare them)

Files follow gpu-rf-pipeline's conventions: .cf32 is interleaved little-endian float32 I/Q,
.f32 raw float32 taps, and every signal has a .json sidecar with its sample rate, length,
generator, seed, and the ground truth a test needs.

    chirp      complex linear FM sweep from -fs/4 to +fs/4, no noise
    rrc        root-raised-cosine pulse: 8 samples per symbol, roll-off 0.35, 8 symbols long
    bpsk       512 random bits, +-1 symbols shaped by rrc, complex noise at 20 dB SNR
    qpsk       512 random Gray-coded symbols (2 bits each), shaped and noised the same way

Matched-filtering bpsk or qpsk with rrc (a second pass of the same pulse) gives a raised-cosine
pulse with no inter-symbol interference: symbol k is at sample k * 8 + len(rrc) - 1.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_RATE = 1_000_000.0
SPS = 8  # samples per symbol
BETA = 0.35  # roll-off
SPAN = 8  # pulse length in symbols
NUM_SYMBOLS = 512
SNR_DB = 20.0

# Gray-coded QPSK: symbol index (two bits b1 b0) -> point. Neighbours differ in one bit.
QPSK_POINTS = np.array([1 + 1j, -1 + 1j, 1 - 1j, -1 - 1j]) / np.sqrt(2.0)


def rrc_taps(sps: int = SPS, beta: float = BETA, span: int = SPAN) -> np.ndarray:
    """Root-raised-cosine impulse response of span * sps + 1 taps, unit energy."""
    t = (np.arange(span * sps + 1) - span * sps / 2) / sps  # in symbol periods
    h = np.empty_like(t)
    for i, ti in enumerate(t):
        if abs(ti) < 1e-12:
            h[i] = 1.0 - beta + 4.0 * beta / np.pi
        elif abs(abs(ti) - 1.0 / (4.0 * beta)) < 1e-12:
            h[i] = (beta / np.sqrt(2.0)) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
                + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta))
            )
        else:
            num = np.sin(np.pi * ti * (1 - beta)) + 4 * beta * ti * np.cos(np.pi * ti * (1 + beta))
            h[i] = num / (np.pi * ti * (1 - (4 * beta * ti) ** 2))
    return h / np.sqrt(np.sum(h**2))


def chirp(n: int = 16384, f0: float = -0.25, f1: float = 0.25) -> np.ndarray:
    """exp(i phase) with frequency rising linearly from f0 to f1 (cycles per sample)."""
    t = np.arange(n)
    phase = 2 * np.pi * (f0 * t + (f1 - f0) / (2 * n) * t**2)
    return np.exp(1j * phase)


def shaped(symbols: np.ndarray, taps: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Symbols upsampled by SPS, filtered by the pulse, plus complex white noise at SNR_DB
    (signal power measured over the whole burst)."""
    up = np.zeros(len(symbols) * SPS, dtype=np.complex128)
    up[::SPS] = symbols
    x = np.convolve(up, taps)[: len(up)]
    noise_power = np.mean(np.abs(x) ** 2) / 10 ** (SNR_DB / 10)
    noise = rng.standard_normal(len(x)) + 1j * rng.standard_normal(len(x))
    return x + noise * np.sqrt(noise_power / 2)


def write(out: Path, name: str, x: np.ndarray, meta: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    np.asarray(x, dtype=np.complex64).view("<f4").tofile(out / f"{name}.cf32")
    meta = {"sample_rate": SAMPLE_RATE, "num_samples": len(x), **meta}
    (out / f"{name}.json").write_text(json.dumps(meta, indent=2) + "\n")


def generate(out: Path) -> None:
    write(
        out,
        "chirp",
        chirp(),
        {
            "generator": "gen_rf_fixtures.chirp",
            "seed": None,
            "f0_hz": -0.25 * SAMPLE_RATE,
            "f1_hz": 0.25 * SAMPLE_RATE,
        },
    )

    taps = rrc_taps()
    out.mkdir(parents=True, exist_ok=True)
    taps.astype("<f4").tofile(out / "rrc.f32")
    (out / "rrc.json").write_text(
        json.dumps(
            {
                "generator": "gen_rf_fixtures.rrc_taps",
                "num_taps": len(taps),
                "samples_per_symbol": SPS,
                "roll_off": BETA,
                "span_symbols": SPAN,
            },
            indent=2,
        )
        + "\n"
    )

    common = {
        "samples_per_symbol": SPS,
        "pulse": "rrc.f32",
        "snr_db": SNR_DB,
        "first_symbol_after_matched_filter": len(taps) - 1,
    }
    rng = np.random.default_rng(1)
    bits = rng.integers(0, 2, NUM_SYMBOLS)
    write(
        out,
        "bpsk",
        shaped(2.0 * bits - 1.0, taps, rng),
        {"generator": "gen_rf_fixtures.bpsk", "seed": 1, "bits": bits.tolist(), **common},
    )

    rng = np.random.default_rng(2)
    symbols = rng.integers(0, 4, NUM_SYMBOLS)
    write(
        out,
        "qpsk",
        shaped(QPSK_POINTS[symbols], taps, rng),
        {
            "generator": "gen_rf_fixtures.qpsk",
            "seed": 2,
            "symbols": symbols.tolist(),
            "constellation": "Gray: 0 -> (1+1j)/sqrt2, 1 -> (-1+1j)/sqrt2, "
            "2 -> (1-1j)/sqrt2, 3 -> (-1-1j)/sqrt2",
            **common,
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=Path, default=ROOT / "tests" / "fixtures" / "rf")
    args = parser.parse_args(argv)
    generate(args.out)
    print(f"wrote the radio fixtures to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
