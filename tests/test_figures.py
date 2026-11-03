"""Smoke test for tools/make_figures.py: it draws every figure from small synthetic results."""

import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

ROOT = Path(__file__).resolve().parent.parent


def load_make_figures():
    spec = importlib.util.spec_from_file_location(
        "make_figures", ROOT / "tools" / "make_figures.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_draws_every_figure_in_both_themes(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    (results / "bench.csv").write_text(
        "# device: test\n"
        "op,impl,n,batch,time_ms,min_ms,gflops,gbps\n"
        "fft_c2c,cufft,64,262144,0.40,0.39,1258.3,671.1\n"
        "fft_c2c,cufft,4096,4096,0.45,0.44,2236.9,596.5\n"
        "scale,ours,16777216,1,0.33,0.32,101.7,813.4\n"
    )
    peaks = {"device": "test", "peak_bandwidth_gbps": 850.0, "peak_fp32_gflops": 30000.0}
    (results / "peaks.json").write_text(json.dumps(peaks))
    figures = tmp_path / "figures"

    assert load_make_figures().main(["--results", str(results), "--figures", str(figures)]) == 0
    assert sorted(p.name for p in figures.iterdir()) == [
        "fft_bandwidth-dark.png",
        "fft_bandwidth.png",
        "roofline-dark.png",
        "roofline.png",
    ]
