"""Radio processing on our FFTs (Task 1.9): windows, the framed spectrum, overlap-save FIR, the
command line, and the chirp / BPSK / QPSK fixtures. Every GPU check runs with cuFFT and with each
of our implementations."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import scipy.signal
import torch

from spectral import rf
from spectral.testing import random_complex, random_real

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures" / "rf"
BACKENDS = rf.FFT_BACKENDS

gpu = pytest.mark.gpu


def cuda(x) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(x)).cuda()


def relative_error(actual, expected) -> float:
    actual = np.asarray(actual, dtype=np.complex128)
    expected = np.asarray(expected, dtype=np.complex128)
    return float(np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1e-30))


def load_fixture(name: str) -> tuple[np.ndarray, dict]:
    return rf.read_cf32(FIXTURES / f"{name}.cf32"), json.loads(
        (FIXTURES / f"{name}.json").read_text()
    )


# --- Windows (CPU) ----------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["hann", "hamming", "blackman"])
@pytest.mark.parametrize("n", [1, 2, 7, 64, 1024])
def test_windows_match_scipy_periodic_windows(name, n):
    expected = scipy.signal.get_window(name, n, fftbins=True)
    np.testing.assert_allclose(rf.window(name, n).numpy(), expected, rtol=0, atol=1e-7)


def test_rect_window_is_all_ones():
    assert torch.equal(rf.window("rect", 5), torch.ones(5))


def test_rejects_an_unknown_window():
    with pytest.raises(ValueError, match="unknown window"):
        rf.window("kaiser", 8)


def test_default_block_size():
    assert rf.default_nfft(1) == 64
    assert rf.default_nfft(63) == 256
    assert rf.default_nfft(1025) == 4096
    assert rf.default_nfft(5000) == 8192


# --- Spectrum ---------------------------------------------------------------------------------


@gpu
@pytest.mark.parametrize("fft_impl", BACKENDS)
@pytest.mark.parametrize("window_name", ["rect", "hann", "blackman"])
@pytest.mark.parametrize("n", [1, 1000, 4096])
def test_spectrum_matches_numpy(rng, fft_impl, window_name, n):
    nfft = 256
    x = random_complex(rng, n)
    count = -(-n // nfft)
    padded = np.zeros(count * nfft, dtype=np.complex128)
    padded[:n] = x
    w = scipy.signal.get_window(window_name, nfft) if window_name != "rect" else np.ones(nfft)
    expected = np.fft.fft(padded.reshape(count, nfft) * w, axis=-1)

    y = rf.spectrum(cuda(x), nfft, window_name=window_name, fft_impl=fft_impl)
    assert y.shape == (count, nfft)
    assert relative_error(y.cpu().numpy(), expected) < 1e-5


# --- FIR by overlap-save ----------------------------------------------------------------------

# (signal length, number of taps): one tap, a signal shorter than the filter, lengths that do not
# fill the last block, a filter longer than the default block's quarter, and a long filter.
FIR_CASES = [(1000, 1), (5, 63), (63, 63), (1000, 63), (4097, 64), (10_000, 1025), (3000, 4096)]


@gpu
@pytest.mark.parametrize("fft_impl", BACKENDS)
@pytest.mark.parametrize(("n", "m"), FIR_CASES)
def test_fir_matches_lfilter(rng, fft_impl, n, m):
    x = random_complex(rng, n)
    taps = random_real(rng, m)
    expected = scipy.signal.lfilter(taps.astype(np.float64), 1.0, x.astype(np.complex128))
    y = rf.fir(cuda(x), cuda(taps), fft_impl=fft_impl)
    assert y.shape == (n,)
    assert relative_error(y.cpu().numpy(), expected) < 2e-5


@gpu
@pytest.mark.parametrize("fft_impl", BACKENDS)
def test_fir_with_complex_taps_rows_and_a_chosen_block_size(rng, fft_impl):
    x = random_complex(rng, (3, 777))
    taps = random_complex(rng, 31)
    expected = scipy.signal.lfilter(taps.astype(np.complex128), 1.0, x.astype(np.complex128))
    y = rf.fir(cuda(x), cuda(taps), nfft=64, fft_impl=fft_impl)
    assert relative_error(y.cpu().numpy(), expected) < 2e-5


@gpu
@pytest.mark.parametrize("fft_impl", BACKENDS)
def test_fir_of_zeros_is_zeros(fft_impl):
    x = torch.zeros(500, dtype=torch.complex64, device="cuda")
    taps = torch.ones(17, device="cuda")
    assert torch.count_nonzero(rf.fir(x, taps, fft_impl=fft_impl)) == 0


@gpu
def test_fir_rejects_a_block_shorter_than_the_filter():
    x = torch.zeros(100, dtype=torch.complex64, device="cuda")
    with pytest.raises(ValueError, match="at least len\\(taps\\) = 65"):
        rf.fir(x, torch.ones(65, device="cuda"), nfft=64)


def test_rejects_an_unknown_backend():
    with pytest.raises(ValueError, match="unknown fft_impl"):
        rf.fft(torch.zeros(4, dtype=torch.complex64), fft_impl="fftw")


# --- Fixtures ---------------------------------------------------------------------------------


def load_generator():
    spec = importlib.util.spec_from_file_location(
        "gen_rf_fixtures", ROOT / "tools" / "gen_rf_fixtures.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fixtures_are_reproducible(tmp_path):
    load_generator().generate(tmp_path)
    committed = sorted(p.name for p in FIXTURES.iterdir())
    assert sorted(p.name for p in tmp_path.iterdir()) == committed
    for name in committed:
        assert (tmp_path / name).read_bytes() == (FIXTURES / name).read_bytes(), name


def test_fixtures_are_small_and_described():
    for path in FIXTURES.glob("*.cf32"):
        x, meta = load_fixture(path.stem)
        assert path.stat().st_size < 1 << 20
        assert meta["num_samples"] == len(x)
        assert {"sample_rate", "generator", "seed"} <= meta.keys()


def test_chirp_frequency_rises_linearly():
    x, meta = load_fixture("chirp")
    inst = np.angle(x[1:] * np.conj(x[:-1])) / (2 * np.pi) * meta["sample_rate"]  # Hz
    ramp = np.linspace(meta["f0_hz"], meta["f1_hz"], len(x), endpoint=False)[:-1]
    assert np.max(np.abs(inst - ramp)) < 0.002 * meta["sample_rate"]


@gpu
@pytest.mark.parametrize("fft_impl", BACKENDS)
def test_chirp_spectrum_peak_follows_the_sweep(fft_impl):
    x, meta = load_fixture("chirp")
    nfft = 256
    spec = rf.spectrum(cuda(x), nfft, window_name="hann", fft_impl=fft_impl).abs().cpu().numpy()
    peaks = np.fft.fftfreq(nfft)[np.argmax(spec, axis=-1)] * meta["sample_rate"]
    centers = (np.arange(len(peaks)) + 0.5) * nfft
    expected = meta["f0_hz"] + (meta["f1_hz"] - meta["f0_hz"]) * centers / len(x)
    assert np.max(np.abs(peaks - expected)) <= 2 * meta["sample_rate"] / nfft


def decode(name: str, fft_impl: str) -> tuple[np.ndarray, dict]:
    """Matched filter on the GPU, then one sample per symbol."""
    x, meta = load_fixture(name)
    taps = rf.read_f32(FIXTURES / meta["pulse"])
    y = rf.fir(cuda(x), cuda(taps), fft_impl=fft_impl).cpu().numpy()
    first, sps = meta["first_symbol_after_matched_filter"], meta["samples_per_symbol"]
    return y[first::sps], meta  # the last few symbols' pulses run past the end of the burst


@gpu
@pytest.mark.parametrize("fft_impl", BACKENDS)
def test_bpsk_bits_are_recovered(fft_impl):
    samples, meta = decode("bpsk", fft_impl)
    bits = np.array(meta["bits"])[: len(samples)]
    assert np.array_equal((samples.real > 0).astype(int), bits)


@gpu
@pytest.mark.parametrize("fft_impl", BACKENDS)
def test_qpsk_symbols_are_recovered(fft_impl):
    samples, meta = decode("qpsk", fft_impl)
    symbols = np.array(meta["symbols"])[: len(samples)]
    decided = (samples.real < 0).astype(int) + 2 * (samples.imag < 0).astype(int)  # Gray map
    assert np.array_equal(decided, symbols)


# --- Command line -----------------------------------------------------------------------------


@gpu
def test_command_line_fir_matches_the_api(tmp_path):
    out = tmp_path / "y.cf32"
    args = ["fir", "--in", str(FIXTURES / "bpsk.cf32"), "--taps", str(FIXTURES / "rrc.f32")]
    assert rf.main([*args, "--out", str(out), "--fft-impl", "stockham_r8"]) == 0
    x, _ = load_fixture("bpsk")
    expected = rf.fir(cuda(x), cuda(rf.read_f32(FIXTURES / "rrc.f32")), fft_impl="stockham_r8")
    np.testing.assert_array_equal(rf.read_cf32(out), expected.cpu().numpy())


@gpu
def test_command_line_spectrum(tmp_path):
    out = tmp_path / "s.cf32"
    args = ["spectrum", "--in", str(FIXTURES / "chirp.cf32"), "--out", str(out)]
    assert rf.main([*args, "--nfft", "512", "--window", "hann"]) == 0
    assert rf.read_cf32(out).size == 16384
