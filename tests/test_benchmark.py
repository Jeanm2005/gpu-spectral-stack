"""Tests for the benchmarking helpers: the work convention, result comparison, CSV reading, and
(on a GPU) that time_cuda measures real kernel time."""

import pytest
import torch

import spectral
from spectral.benchmark import compare, fft_c2c_work, read_results, time_cuda


def test_fft_work_follows_the_5_n_log2_n_convention():
    flops, nbytes = fft_c2c_work(1024, 3)
    assert flops == 5 * 1024 * 10 * 3
    assert nbytes == 16 * 1024 * 3


def test_compare_flags_only_changes_beyond_the_threshold():
    old = {("fft", "a", 64, 1): 1.0, ("fft", "b", 64, 1): 1.0, ("fft", "c", 64, 1): 1.0}
    new = {("fft", "a", 64, 1): 1.05, ("fft", "b", 64, 1): 1.2, ("fft", "c", 64, 1): 0.5}
    lines, regressed = compare(old, new, threshold=0.10)
    assert regressed
    assert [line.split()[1] for line in lines if "SLOWER" in line] == ["b"]
    assert [line.split()[1] for line in lines if "faster" in line] == ["c"]


def test_compare_lists_unmatched_rows_without_failing():
    lines, regressed = compare({("fft", "a", 64, 1): 1.0}, {("fft", "a", 128, 1): 1.0}, 0.10)
    assert not regressed
    assert any(line.startswith("only in old") for line in lines)
    assert any(line.startswith("only in new") for line in lines)


def test_read_results_skips_comment_lines(tmp_path):
    path = tmp_path / "bench.csv"
    path.write_text(
        "# device: test\n"
        "op,impl,n,batch,time_ms,min_ms,gflops,gbps\n"
        "fft_c2c,cufft,64,262144,0.250000,0.240000,100.0,800.0\n"
    )
    assert read_results(path) == {("fft_c2c", "cufft", 64, 262144): 0.25}


def test_time_cuda_rejects_a_bad_configuration():
    with pytest.raises(ValueError, match="rounds"):
        time_cuda(lambda: None, rounds=0)


@pytest.mark.gpu
def test_time_cuda_measures_real_kernel_time():
    x = torch.randn(1 << 24, dtype=torch.complex64, device="cuda")
    t = time_cuda(lambda: spectral.scale(x, 0.5))
    assert 0 < t.min_ms <= t.median_ms <= t.max_ms
    # 16 bytes per point at a believable rate. A timing bug (not waiting for the GPU) shows up
    # as an impossible bandwidth of many terabytes per second.
    gbps = 16 * (1 << 24) / (t.median_ms * 1e6)
    assert 10 < gbps < 2000
