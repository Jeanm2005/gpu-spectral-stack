"""The training harness (Task 3.1): configs, the trajectory split, tracking, checkpoints, resume,
and reproducibility, on a tiny dataset made by the NumPy solver. CPU, except one GPU smoke test."""

import json
import math

import numpy as np
import pytest

from spectral.flow.generate import DatasetConfig
from spectral.flow.store import write_dataset
from spectral.fno.config import Config, load_config
from spectral.fno.tracking import JsonTracker
from spectral.fno.train import split_samples, train

DATA = DatasetConfig(
    n=16,
    dt=0.01,
    record_every=2,
    frames=8,
    num_samples=8,
    batch_size=8,
    viscosity_min=1e-2,
    viscosity_max=2e-2,
    backend="numpy",
)


@pytest.fixture(scope="module")
def data_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("fno") / "flow.zarr"
    write_dataset(path, DATA, samples_per_shard=4)
    return path


def tiny_config(data_path, tmp_path, **train) -> Config:
    return Config.from_dict(
        {
            "run_name": "tiny",
            "seed": 0,
            "data": {
                "path": str(data_path),
                "inputs": 3,
                "outputs": 1,
                "test_samples": 2,
                "num_workers": 0,
            },
            "model": {"modes": 4, "width": 8, "layers": 2, "projection": 16},
            "train": {"epochs": 2, "batch_size": 8, "device": "cpu", **train},
            "tracking": {"backend": "json", "uri": str(tmp_path / "runs")},
            "checkpoint": {"dir": str(tmp_path / "ckpt")},
        }
    )


def run_dirs(tmp_path):
    return sorted((tmp_path / "runs").iterdir())


def logged_metrics(run_dir):
    return [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]


def test_split_holds_out_whole_trajectories():
    train_ids, test_ids = split_samples(100, 20, seed=3)
    assert len(train_ids) == 80 and len(test_ids) == 20
    assert not set(train_ids) & set(test_ids)
    np.testing.assert_array_equal(np.sort(np.concatenate([train_ids, test_ids])), np.arange(100))
    with pytest.raises(ValueError):
        split_samples(10, 10, seed=0)


def test_config_file_and_overrides(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("run_name: a\ntrain:\n  epochs: 7\n")
    config = load_config(path, ["train.lr=3.0e-4", "model.width=16", "tracking.uri=null"])
    assert (config.run_name, config.train.epochs, config.train.lr) == ("a", 7, 3e-4)
    assert config.model.width == 16 and config.tracking.uri is None
    assert config.flat()["train.epochs"] == 7
    with pytest.raises(ValueError, match="unknown keys in train"):
        load_config(path, ["train.epoch=3"])


def test_training_runs_records_and_checkpoints(data_path, tmp_path):
    result = train(tiny_config(data_path, tmp_path), JsonTracker(tmp_path / "runs"))
    assert math.isfinite(result["best_test_rel_l2"])

    (run,) = run_dirs(tmp_path)
    metrics = logged_metrics(run)
    assert [m["step"] for m in metrics] == [0, 1]
    assert all(math.isfinite(m["train_rel_l2"] + m["test_rel_l2"]) for m in metrics)
    params = json.loads((run / "params.json").read_text())
    assert params["data.inputs"] == 3 and params["data.train_samples"] == 6
    assert len(params["dataset.config_hash"]) > 0
    assert json.loads((run / "run.json").read_text())["status"] == "FINISHED"
    assert (run / "artifacts" / "best.pt").exists()
    assert (tmp_path / "ckpt" / "tiny" / "latest.pt").exists()


def test_training_is_reproducible(data_path, tmp_path):
    first = train(tiny_config(data_path, tmp_path / "a"), JsonTracker(tmp_path / "a" / "runs"))
    second = train(tiny_config(data_path, tmp_path / "b"), JsonTracker(tmp_path / "b" / "runs"))
    assert first["train_rel_l2"] == second["train_rel_l2"]
    assert first["test_rel_l2"] == second["test_rel_l2"]


def test_resume_continues_after_the_last_epoch(data_path, tmp_path):
    train(tiny_config(data_path, tmp_path, epochs=1), JsonTracker(tmp_path / "runs"))
    train(tiny_config(data_path, tmp_path, epochs=3, resume=True), JsonTracker(tmp_path / "runs"))
    first, second = run_dirs(tmp_path)
    assert [m["step"] for m in logged_metrics(first)] == [0]
    assert [m["step"] for m in logged_metrics(second)] == [1, 2]
    assert json.loads((second / "params.json").read_text())["resumed_from_epoch"] == 1


def test_a_bad_config_fails_before_anything_is_logged(data_path, tmp_path):
    config = tiny_config(data_path, tmp_path)
    config.model.backend = "nope"  # fails when the model is built, before the run starts
    with pytest.raises(ValueError):
        train(config, JsonTracker(tmp_path / "runs"))
    assert not (tmp_path / "runs").exists() or not run_dirs(tmp_path)


def test_mlflow_tracker_records_a_run(data_path, tmp_path):
    mlflow = pytest.importorskip("mlflow")
    from spectral.fno.tracking import MlflowTracker

    uri = f"sqlite:///{tmp_path / 'db' / 'mlflow.db'}"
    tracker = MlflowTracker("tests", uri)
    run_id = tracker.start("r", {"a": 1, "b": None})
    tracker.log_metrics({"loss": 0.5}, step=0)
    tracker.finish()
    run = mlflow.tracking.MlflowClient(uri).get_run(run_id)
    assert run.data.params == {"a": "1", "b": "None"} and run.data.metrics == {"loss": 0.5}
    assert run.info.status == "FINISHED"


@pytest.mark.gpu
def test_training_on_the_gpu(data_path, tmp_path):
    result = train(tiny_config(data_path, tmp_path, device="cuda"), JsonTracker(tmp_path / "runs"))
    assert math.isfinite(result["best_test_rel_l2"]) and result["peak_memory_mb"] > 0
