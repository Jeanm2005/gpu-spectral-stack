"""Batched trajectory generation (Task 2.3): configs and their hashes, per-sample seeds, the
initial-condition prior, and that a dataset does not depend on how it is batched."""

import dataclasses

import numpy as np
import pytest

from spectral.flow import Grid
from spectral.flow import reference as ref
from spectral.flow.generate import (
    DatasetConfig,
    config_hash,
    draw_samples,
    gaussian_random_field,
    generate,
    sample_seed,
    sample_viscosity,
)

TINY = DatasetConfig(n=16, dt=0.01, record_every=5, frames=4, num_samples=5, batch_size=2)


def collect(config):
    batches = list(generate(config))
    return {
        "indices": np.concatenate([b.indices for b in batches]),
        "seeds": np.concatenate([b.seeds for b in batches]),
        "viscosity": np.concatenate([b.viscosity for b in batches]),
        "vorticity": np.concatenate([b.vorticity for b in batches]),
    }


# --- Configs and hashes -----------------------------------------------------------------------


def test_hash_is_stable_and_ignores_the_batch_size():
    a = config_hash(TINY)
    assert a == config_hash(DatasetConfig.from_dict(TINY.as_dict()))
    assert len(a) == 64
    assert config_hash(dataclasses.replace(TINY, batch_size=5)) == a


@pytest.mark.parametrize(
    "change", [{"seed": 1}, {"n": 32}, {"dt": 0.02}, {"viscosity_max": 2e-3}, {"pruned": False}]
)
def test_hash_changes_with_anything_that_changes_the_data(change):
    assert config_hash(dataclasses.replace(TINY, **change)) != config_hash(TINY)


def test_identity_records_the_schema_version():
    assert TINY.identity()["schema_version"] >= 1
    assert "batch_size" not in TINY.identity()


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"n": 48}, "power of two"),
        ({"frames": 0}, "frames >= 1"),
        ({"viscosity_min": 2e-3, "viscosity_max": 1e-3}, "viscosity_min <= viscosity_max"),
        ({"backend": "tpu"}, "backend"),
    ],
)
def test_rejects_invalid_configs(change, message):
    with pytest.raises(ValueError, match=message):
        dataclasses.replace(TINY, **change)


def test_from_dict_rejects_unknown_keys():
    with pytest.raises(ValueError, match="unknown dataset config keys"):
        DatasetConfig.from_dict({"n": 16, "reynolds": 100})


def test_times_of_the_recorded_frames():
    config = dataclasses.replace(TINY, spinup_steps=10)
    np.testing.assert_allclose(config.times, [0.1, 0.15, 0.2, 0.25])


# --- Seeds, viscosity, initial conditions -----------------------------------------------------


def test_sample_seeds_are_distinct_and_independent_of_other_samples():
    seeds = [sample_seed(TINY, i) for i in range(100)]
    assert len(set(seeds)) == 100
    bigger = dataclasses.replace(TINY, num_samples=1000)
    assert sample_seed(bigger, 7) == seeds[7]


def test_viscosity_is_log_uniform_in_its_range():
    config = dataclasses.replace(TINY, viscosity_min=1e-4, viscosity_max=1e-2)
    rng = np.random.default_rng(0)
    nus = np.array([sample_viscosity(config, rng) for _ in range(4000)])
    assert nus.min() >= 1e-4 and nus.max() <= 1e-2
    logs = np.log10(nus)
    assert logs.mean() == pytest.approx(-3.0, abs=0.05)  # the middle of [-4, -2]
    assert np.histogram(logs, bins=4, range=(-4, -2))[0].min() > 900  # flat


def test_a_fixed_viscosity_range_gives_that_viscosity():
    assert sample_viscosity(TINY, np.random.default_rng(0)) == TINY.viscosity_min


def test_initial_conditions_have_zero_mean_and_the_configured_rms():
    w = gaussian_random_field(TINY, np.random.default_rng(1))
    assert abs(w.mean()) < 1e-12
    assert w.std() == pytest.approx(TINY.ic_rms)


def test_initial_condition_spectrum_follows_the_prior():
    config = dataclasses.replace(TINY, n=64)
    grid = Grid(64)
    rng = np.random.default_rng(2)
    power = np.mean(
        [np.abs(ref.rfft2(gaussian_random_field(config, rng))) ** 2 for _ in range(200)], axis=0
    )
    k = np.sqrt(grid.k2)
    low, high = (np.abs(k - 2) < 0.5), (np.abs(k - 16) < 0.5)
    measured = power[high].mean() / power[low].mean()
    prior = ((16**2 + 49) / (2**2 + 49)) ** -2.5  # (|k|^2 + tau^2)^(-alpha), power
    assert measured == pytest.approx(prior, rel=0.25)


def test_draw_samples_is_deterministic():
    a = draw_samples(TINY, np.arange(3))
    b = draw_samples(TINY, np.arange(3))
    for x, y in zip(a, b, strict=True):
        np.testing.assert_array_equal(x, y)


# --- Generation -------------------------------------------------------------------------------


def numpy_config(**change):
    return dataclasses.replace(TINY, backend="numpy", **change)


def test_numpy_backend_produces_every_sample_in_order():
    data = collect(numpy_config())
    assert data["indices"].tolist() == [0, 1, 2, 3, 4]
    assert data["vorticity"].shape == (5, 4, 16, 16)
    assert data["vorticity"].dtype == np.float32
    assert np.isfinite(data["vorticity"]).all()


def test_numpy_backend_does_not_depend_on_the_batch_size():
    a = collect(numpy_config(batch_size=1))
    b = collect(numpy_config(batch_size=5))
    np.testing.assert_array_equal(a["vorticity"], b["vorticity"])
    np.testing.assert_array_equal(a["seeds"], b["seeds"])


def test_a_range_of_samples_matches_the_full_run():
    full = collect(numpy_config())
    part = list(generate(numpy_config(), start=3, stop=5))
    np.testing.assert_array_equal(
        np.concatenate([b.vorticity for b in part]), full["vorticity"][3:]
    )


def test_spinup_runs_before_the_first_frame():
    data = collect(numpy_config(spinup_steps=10))
    _, _, w0 = draw_samples(TINY, np.arange(5))
    assert not np.allclose(data["vorticity"][:, 0], w0, atol=1e-3)


@pytest.mark.gpu
def test_gpu_backend_does_not_depend_on_the_batch_size():
    a = collect(dataclasses.replace(TINY, batch_size=1))
    b = collect(dataclasses.replace(TINY, batch_size=5))
    np.testing.assert_allclose(a["vorticity"], b["vorticity"], rtol=0, atol=1e-5)


@pytest.mark.gpu
@pytest.mark.parametrize(("fft_impl", "pruned"), [("stockham_r8", True), ("cufft", False)])
def test_gpu_backend_agrees_with_the_numpy_backend(fft_impl, pruned):
    config = dataclasses.replace(TINY, n=32, viscosity_max=5e-3, fft_impl=fft_impl, pruned=pruned)
    gpu = collect(config)["vorticity"]
    cpu = collect(dataclasses.replace(config, backend="numpy"))["vorticity"]
    assert np.linalg.norm(gpu - cpu) / np.linalg.norm(cpu) < 1e-4
