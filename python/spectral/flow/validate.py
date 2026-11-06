"""Dataset validation (Task 2.5): a dataset that fails any check fails the pipeline.

    python -m spectral.flow.validate data/train.zarr --report out/validation.json

Checks, each with its threshold in Thresholds:

    complete      every sample was written
    finite        no NaN or infinity anywhere
    bounded       max |omega| stays below max_abs_vorticity (blow-up)
    cfl           max |u| dt / dx of every frame stays below max_cfl: past about 1 the explicit
                  part of the time stepping is unstable, and a run can stay finite while its
                  numbers mean nothing
    energy        the energy budget closes between consecutive frames: the change in E agrees
                  with dE/dt = <u.F> - 2 nu Z - 2 alpha E integrated over the interval
                  (trapezoid rule), to max_energy_residual of the trajectory's mean energy.
                  Catches instability and solver errors that stay inside the bounds
    parameters    viscosities inside the configured range and, from 20 samples on, consistent
                  with a log-uniform distribution (Kolmogorov-Smirnov, p >= min_ks_pvalue)
    provenance    every stored seed is the one flow.generate derives for that sample, and the
                  time axis matches the config

The report (JSON) holds every check's verdict and the per-trajectory statistics behind it:
energy and enstrophy over time, max |omega|, max CFL, budget residual, viscosity.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import scipy.stats

from . import reference
from .generate import sample_seed
from .params import Flow, Grid
from .store import Dataset, open_dataset


@dataclass(frozen=True)
class Thresholds:
    max_abs_vorticity: float = 1e3
    max_cfl: float = 1.0
    max_energy_residual: float = 0.05
    min_ks_pvalue: float = 1e-3


def trajectory_stats(w: np.ndarray, viscosity: float, ds: Dataset) -> dict:
    """Diagnostics of one trajectory (frames, n, n): its energy budget residual, max CFL, ..."""
    config = ds.config
    flow = Flow(
        viscosity=viscosity,
        drag=config.drag,
        forcing_amplitude=config.forcing_amplitude,
        forcing_wavenumber=config.forcing_wavenumber,
    )
    w = w.astype(np.float64)
    d = reference.diagnostics(w, flow)
    rate, _ = reference.budget_rates(w, flow)
    interval = config.dt * config.record_every
    residual = np.diff(d["energy"]) - 0.5 * interval * (rate[1:] + rate[:-1])
    scale = max(float(d["energy"].mean()), 1e-12)

    grid = Grid(config.n)
    solver = reference.Solver(grid, flow, config.dt)
    u_hat, v_hat = solver.velocity_hat(reference.to_spectral(w, grid))
    speed = np.sqrt(reference.irfft2(u_hat, config.n) ** 2 + reference.irfft2(v_hat, config.n) ** 2)
    dx = 2.0 * np.pi / config.n
    return {
        "viscosity": viscosity,
        "max_abs_vorticity": float(np.abs(w).max()),
        "max_cfl": float(speed.max() * config.dt / dx),
        "energy_residual": float(np.abs(residual).max() / scale) if residual.size else 0.0,
        "energy": d["energy"].tolist(),
        "enstrophy": d["enstrophy"].tolist(),
    }


def validate(ds: Dataset, thresholds: Thresholds | None = None) -> dict:
    """Runs every check; returns the report. report["passed"] is the overall verdict."""
    thresholds = Thresholds() if thresholds is None else thresholds
    config = ds.config
    checks: dict[str, dict] = {}

    missing = ds.missing()
    checks["complete"] = {"passed": missing.size == 0, "missing_samples": missing.tolist()}

    viscosity = ds.viscosity[:]
    stats, nonfinite = [], []
    missing_set = set(missing.tolist())
    for i in range(config.num_samples):
        if i in missing_set:
            stats.append(None)
            continue
        w = ds.vorticity[i]
        if not np.isfinite(w).all():
            nonfinite.append(i)
            stats.append(None)
            continue
        stats.append(trajectory_stats(w, float(viscosity[i]), ds))
    checks["finite"] = {"passed": not nonfinite, "nonfinite_samples": nonfinite}

    def over(key, limit):
        return [i for i, s in enumerate(stats) if s is not None and s[key] > limit]

    bad = over("max_abs_vorticity", thresholds.max_abs_vorticity)
    checks["bounded"] = {"passed": not bad, "samples": bad, "limit": thresholds.max_abs_vorticity}
    bad = over("max_cfl", thresholds.max_cfl)
    checks["cfl"] = {"passed": not bad, "samples": bad, "limit": thresholds.max_cfl}
    bad = over("energy_residual", thresholds.max_energy_residual)
    checks["energy"] = {
        "passed": not bad,
        "samples": bad,
        "limit": thresholds.max_energy_residual,
    }

    written = ~np.isin(np.arange(config.num_samples), missing)
    nus = viscosity[written]
    lo, hi = config.viscosity_min, config.viscosity_max
    in_range = bool(np.all((nus >= lo * (1 - 1e-12)) & (nus <= hi * (1 + 1e-12))))
    parameters = {"in_range": in_range, "min": float(nus.min()), "max": float(nus.max())}
    passed = in_range
    if hi > lo and nus.size >= 20:
        u = (np.log(nus) - np.log(lo)) / (np.log(hi) - np.log(lo))
        pvalue = float(scipy.stats.kstest(u, "uniform").pvalue)
        parameters["ks_pvalue_log_uniform"] = pvalue
        passed = passed and pvalue >= thresholds.min_ks_pvalue
    parameters["histogram_log10"] = np.histogram(np.log10(nus), bins=10)[0].tolist()
    checks["parameters"] = {"passed": passed, **parameters}

    seeds = ds.seed[:]
    wrong = [
        i
        for i in range(config.num_samples)
        if written[i] and int(seeds[i]) != sample_seed(config, i)
    ]
    times_ok = bool(np.allclose(ds.time[:], config.times))
    checks["provenance"] = {"passed": not wrong and times_ok, "wrong_seeds": wrong}
    checks["provenance"]["times_match_config"] = times_ok

    return {
        "passed": all(c["passed"] for c in checks.values()),
        "config_hash": ds.attrs["config_hash"],
        "git_commit": ds.attrs["git_commit"],
        "thresholds": asdict(thresholds),
        "checks": checks,
        "trajectories": stats,
    }


def summary(report: dict) -> str:
    lines = [f"dataset {report['config_hash'][:12]}: {'PASSED' if report['passed'] else 'FAILED'}"]
    for name, check in report["checks"].items():
        detail = {k: v for k, v in check.items() if k not in ("passed", "histogram_log10")}
        lines.append(f"  {'ok  ' if check['passed'] else 'FAIL'} {name}: {detail}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m spectral.flow.validate",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--report", type=Path, help="write the full report as JSON")
    for f in Thresholds.__dataclass_fields__.values():
        parser.add_argument(f"--{f.name.replace('_', '-')}", type=float, default=f.default)
    args = parser.parse_args(argv)

    thresholds = Thresholds(
        **{name: getattr(args, name) for name in Thresholds.__dataclass_fields__}
    )
    try:
        ds = open_dataset(args.dataset)
    except ValueError as e:  # schema violations
        print(f"FAILED: {e}", file=sys.stderr)
        return 1
    report = validate(ds, thresholds)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=1) + "\n")
    print(summary(report))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
