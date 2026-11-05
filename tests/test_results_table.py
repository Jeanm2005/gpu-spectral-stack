"""Tests for tools/results_table.py: the tables it builds from benchmark CSVs, and the README."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

CSV = """# device: test
op,impl,n,batch,time_ms,min_ms,max_ms,gflops,gbps
fft_c2c,cufft,64,4,1.0,1.0,1.0,1,1
fft_c2c,naive,64,4,4.0,4.0,4.0,1,1
fft_c2c,stockham_r8,64,4,0.8,0.8,0.8,1,1
fft_c2c,cufft,128,2,2.0,2.0,2.0,1,1
fft_c2c,stockham_r8,128,2,2.5,2.5,2.5,1,1
rfft2_modes,cufft+gather,64,1,3.0,3.0,3.0,1,1
rfft2_modes,stockham_r8,64,1,1.0,1.0,1.0,1,1
"""


@pytest.fixture
def tool():
    spec = importlib.util.spec_from_file_location(
        "results_table", ROOT / "tools" / "results_table.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def results(tmp_path):
    (tmp_path / "bench.csv").write_text(CSV)
    return tmp_path


def test_best_table_picks_our_fastest_and_compares_with_cufft(tool, results):
    lines = tool.best_table(tool.load(results / "bench.csv"), "fft_c2c", "n")
    assert lines[2] == "| 64 | 1.0000 | stockham_r8 | 0.8000 | 125 % |"
    assert lines[3] == "| 128 | 2.0000 | stockham_r8 | 2.5000 | 80 % |"


def test_pruned_rows_compare_against_cufft_plus_gather(tool, results):
    lines = tool.best_table(tool.load(results / "bench.csv"), "rfft2_modes", "n")
    assert lines[2] == "| 64 | 3.0000 | stockham_r8 | 1.0000 | 300 % |"


def test_every_impl_table_has_a_column_per_implementation(tool, results):
    lines = tool.every_impl_table(tool.load(results / "bench.csv"), "fft_c2c")
    assert lines[0] == "| n | naive | stockham_r8 |"
    assert lines[2] == "| 64 | 25 % | 125 % |"
    assert lines[3] == "| 128 |  | 80 % |"  # naive has no row at 128


def test_updates_the_readme_between_the_markers_and_checks_it(tool, results, tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text(f"intro\n{tool.BEGIN}\nold\n{tool.END}\noutro\n")
    args = ["--results", str(results), "--readme", str(readme)]

    assert tool.main([*args, "--check"]) == 1
    assert tool.main(args) == 0
    text = readme.read_text()
    assert text.startswith("intro\n") and text.endswith("outro\n")
    assert "| 64 | 1.0000 | stockham_r8 | 0.8000 | 125 % |" in text
    assert "old" not in text
    assert tool.main([*args, "--check"]) == 0


def test_the_committed_readme_matches_the_committed_results(tool):
    assert tool.main(["--check"]) == 0, "run python tools/results_table.py"
