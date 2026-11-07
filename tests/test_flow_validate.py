"""Dataset validation (Task 2.5): a healthy dataset passes, and each kind of bad dataset fails the
check meant for it. CPU only (the NumPy backend)."""

import dataclasses
import json

import numpy as np
import pytest

from spectral.flow.generate import DatasetConfig
from spectral.flow.store import create, open_dataset, write_dataset
from spectral.flow.validate import main, validate

# Viscous enough that a 32 x 32 grid resolves the dissipation range (the `resolved` check).
CONFIG = DatasetConfig(
    n=32,
    dt=0.01,
    record_every=5,
    frames=6,
    num_samples=24,
    batch_size=24,
    viscosity_min=5e-2,
    viscosity_max=1e-1,
    backend="numpy",
)


@pytest.fixture(scope="module")
def good(tmp_path_factory):
    return write_dataset(tmp_path_factory.mktemp("good") / "d.zarr", CONFIG)


@pytest.fixture
def copy(tmp_path, good):
    """A writable copy of the healthy dataset, to break."""
    ds = create(tmp_path / "d.zarr", CONFIG)
    ds.vorticity[:] = good.vorticity[:]
    ds.viscosity[:] = good.viscosity[:]
    ds.seed[:] = good.seed[:]
    ds.written[:] = True
    return ds


def failed(report):
    return sorted(name for name, c in report["checks"].items() if not c["passed"])


def test_a_healthy_dataset_passes(good):
    report = validate(good)
    assert report["passed"], failed(report)
    assert report["checks"]["parameters"]["ks_pvalue_log_uniform"] > 1e-3
    t = report["trajectories"][0]
    assert len(t["energy"]) == CONFIG.frames and t["max_cfl"] < 1
    assert max(s["energy_residual"] for s in report["trajectories"]) < 0.01


def test_a_missing_sample_fails_completeness(copy):
    copy.written[3] = False
    report = validate(copy)
    assert failed(report) == ["complete"]
    assert report["checks"]["complete"]["missing_samples"] == [3]


def test_nan_fails_finiteness(copy):
    w = copy.vorticity[2]
    w[4, 1, 1] = np.nan
    copy.vorticity[2] = w
    assert failed(validate(copy)) == ["finite"]


def test_huge_values_fail_the_bound_and_the_cfl_check(copy):
    copy.vorticity[5] = copy.vorticity[5] * 1e4
    report = validate(copy)
    assert {"bounded", "cfl"} <= set(failed(report))
    assert report["checks"]["bounded"]["samples"] == [5]


def test_a_corrupted_frame_fails_the_energy_budget(copy):
    w = copy.vorticity[7]
    w[3] *= 1.3  # one frame with 69 % more energy than its neighbours allow
    copy.vorticity[7] = w
    report = validate(copy)
    assert failed(report) == ["energy"]
    assert report["checks"]["energy"]["samples"] == [7]


def test_a_viscosity_out_of_range_fails_the_parameter_check(copy):
    copy.viscosity[0] = 1.0
    assert "parameters" in failed(validate(copy))


def test_viscosities_that_are_not_log_uniform_fail(copy):
    copy.viscosity[:] = np.full(CONFIG.num_samples, 1.9e-2)  # all at the top of the range
    report = validate(copy)
    assert "parameters" in failed(report)
    assert report["checks"]["parameters"]["ks_pvalue_log_uniform"] < 1e-3


def test_a_foreign_seed_fails_provenance(copy):
    copy.seed[4] = 12345
    report = validate(copy)
    assert failed(report) == ["provenance"]
    assert report["checks"]["provenance"]["wrong_seeds"] == [4]


def test_an_under_resolved_dataset_fails_the_resolution_check(tmp_path):
    # Stable, finite, and budget-consistent, but too turbulent for 16 x 16: enstrophy piles up
    # at the dealiasing cutoff.
    coarse = dataclasses.replace(
        CONFIG, n=16, num_samples=4, batch_size=4, viscosity_min=1e-3, viscosity_max=1e-3
    )
    report = validate(write_dataset(tmp_path / "d.zarr", coarse))
    assert failed(report) == ["resolved"]
    assert report["checks"]["resolved"]["worst"] < 1


def test_an_unstable_time_step_is_caught(tmp_path):
    # Far past the CFL limit: the run either blows up or stays finite but meaningless.
    unstable = dataclasses.replace(
        CONFIG, n=32, dt=0.5, num_samples=4, batch_size=4, viscosity_min=1e-3, viscosity_max=1e-3
    )
    with np.errstate(all="ignore"):
        ds = write_dataset(tmp_path / "d.zarr", unstable)
    report = validate(ds)
    assert not report["passed"]
    assert set(failed(report)) & {"finite", "bounded", "cfl", "energy"}


def test_command_line_exit_status_and_report(tmp_path, good, copy):
    out = tmp_path / "report.json"
    assert main([str(good.group.store.root), "--report", str(out)]) == 0
    assert json.loads(out.read_text())["passed"] is True

    copy.written[0] = False
    assert main([str(tmp_path / "d.zarr")]) == 1


def test_command_line_fails_on_bad_metadata(tmp_path, copy):
    copy.group.attrs["resolution"] = 99
    assert main([str(tmp_path / "d.zarr")]) == 1


def test_thresholds_can_be_tightened(good):
    from spectral.flow.validate import Thresholds

    report = validate(open_dataset(good.group.store.root), Thresholds(max_cfl=1e-6))
    assert failed(report) == ["cfl"]
