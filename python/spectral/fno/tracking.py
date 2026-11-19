"""Experiment tracking (Task 3.1): one interface, two backends.

    MlflowTracker   MLflow runs: parameters, per-epoch metrics, and artifacts (the best
                    checkpoint, the normalization), browsable with `mlflow ui`
    JsonTracker     the same records as plain files under runs/<run id>/, with no dependency:
                    for tests, CI, and machines without MLflow

Both record exactly what the training loop gives them, so a run can be compared across backends.
"""

from __future__ import annotations

import datetime
import json
import shutil
from pathlib import Path
from typing import Protocol


class Tracker(Protocol):
    def start(self, run_name: str, params: dict) -> str: ...  # returns the run id
    def log_metrics(self, metrics: dict, step: int) -> None: ...
    def log_artifact(self, path: str | Path) -> None: ...
    def finish(self, status: str = "FINISHED") -> None: ...  # or "FAILED"


class JsonTracker:
    """runs/<timestamp>-<run name>/: params.json, metrics.jsonl (one line per step), run.json
    (name, start and end time, status), and artifacts/."""

    def __init__(self, root: str | Path = "runs"):
        self.root = Path(root)
        self.dir: Path | None = None
        self.info: dict = {}

    def start(self, run_name: str, params: dict) -> str:
        now = datetime.datetime.now(datetime.timezone.utc)
        run_id = f"{now:%Y%m%dT%H%M%S%fZ}-{run_name}"
        self.dir = self.root / run_id
        (self.dir / "artifacts").mkdir(parents=True)
        (self.dir / "params.json").write_text(json.dumps(params, indent=2, sort_keys=True))
        self.info = {"run_id": run_id, "name": run_name, "start_utc": now.isoformat()}
        self._write_info("RUNNING")
        return run_id

    def log_metrics(self, metrics: dict, step: int) -> None:
        with (self._dir() / "metrics.jsonl").open("a") as f:
            f.write(json.dumps({"step": step, **metrics}) + "\n")

    def log_artifact(self, path: str | Path) -> None:
        shutil.copy2(path, self._dir() / "artifacts" / Path(path).name)

    def finish(self, status: str = "FINISHED") -> None:
        if self.dir is not None:
            self.info["end_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            self._write_info(status)

    def _dir(self) -> Path:
        if self.dir is None:
            raise RuntimeError("start() the run first")
        return self.dir

    def _write_info(self, status: str) -> None:
        (self._dir() / "run.json").write_text(json.dumps({**self.info, "status": status}, indent=2))


DEFAULT_MLFLOW_URI = "sqlite:///runs/mlflow.db"


class MlflowTracker:
    """An MLflow run in `experiment`. uri: the tracking URI, a server (http://...) or a database
    (sqlite:///path.db). None means the MLFLOW_TRACKING_URI environment variable if it is set,
    else DEFAULT_MLFLOW_URI. MLflow 3 put the old ./mlruns file store in maintenance mode, so the
    local default is a SQLite file; its artifacts go to ./mlartifacts beside it."""

    def __init__(self, experiment: str = "fno", uri: str | None = None):
        import os

        import mlflow  # imported only when this backend is used

        self.mlflow = mlflow
        uri = uri or os.environ.get("MLFLOW_TRACKING_URI") or DEFAULT_MLFLOW_URI
        if uri.startswith("sqlite:///"):
            Path(uri.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(experiment)
        self.active = False

    def start(self, run_name: str, params: dict) -> str:
        run = self.mlflow.start_run(run_name=run_name)
        self.active = True
        self.mlflow.log_params({k: "None" if v is None else v for k, v in params.items()})
        return run.info.run_id

    def log_metrics(self, metrics: dict, step: int) -> None:
        self.mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, path: str | Path) -> None:
        self.mlflow.log_artifact(str(path))

    def finish(self, status: str = "FINISHED") -> None:
        if self.active:
            self.mlflow.end_run(status=status)
            self.active = False


def make_tracker(backend: str, experiment: str, uri: str | None) -> Tracker:
    if backend == "mlflow":
        return MlflowTracker(experiment, uri)
    if backend == "json":
        return JsonTracker(uri or "runs")
    raise ValueError(f"unknown tracking backend {backend!r}: use mlflow or json")
