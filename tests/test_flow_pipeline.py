"""The data pipeline's stages (Task 2.6), and the parameter files DVC runs them with. CPU only:
CI runs these, and runs the whole pipeline through `dvc repro` on configs/ci-params.yaml."""

from pathlib import Path

import pytest
import yaml

from spectral.flow.generate import DatasetConfig
from spectral.flow.pipeline import load_params, main
from spectral.flow.store import open_dataset
from spectral.flow.validate import Thresholds

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("path", ["params.yaml", "configs/ci-params.yaml"])
def test_parameter_files_are_valid(path):
    params = load_params(ROOT / path)
    DatasetConfig.from_dict(params["dataset"])  # raises on unknown keys or bad values
    Thresholds(**params["validation"])
    assert set(params["layout"]) <= {"chunk_frames", "samples_per_shard", "compression"}


def test_ci_parameters_run_without_a_gpu():
    assert load_params(ROOT / "configs/ci-params.yaml")["dataset"]["backend"] == "numpy"


def test_rejects_an_unknown_section(tmp_path):
    (tmp_path / "p.yaml").write_text("dataset: {}\nextras: {}\n")
    with pytest.raises(ValueError, match="unknown sections"):
        load_params(tmp_path / "p.yaml")


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    """All three stages on configs/ci-params.yaml, shrunk further."""
    tmp = tmp_path_factory.mktemp("pipeline")
    params = load_params(ROOT / "configs/ci-params.yaml")
    params["dataset"].update(n=16, num_samples=2, batch_size=2, frames=4, spinup_steps=10)
    params["dataset"].update(viscosity_min=0.1, viscosity_max=0.1)
    (tmp / "params.yaml").write_text(yaml.safe_dump(params))
    data, report = tmp / "flow.zarr", tmp / "reports" / "validation.json"
    codes = [
        main(["generate", "--params", str(tmp / "params.yaml"), "--out", str(data)]),
        main(["validate", "--params", str(tmp / "params.yaml"), "--data", str(data),
              "--report", str(report)]),
        main(["report", "--data", str(data), "--validation", str(report),
              "--out", str(tmp / "reports")]),
    ]  # fmt: skip
    return tmp, codes


def test_the_stages_run_end_to_end(run):
    tmp, codes = run
    assert codes == [0, 0, 0]
    assert open_dataset(tmp / "flow.zarr").complete
    assert (tmp / "reports" / "samples.png").stat().st_size > 1000


def test_the_dataset_card_describes_the_dataset(run):
    tmp, _ = run
    ds = open_dataset(tmp / "flow.zarr")
    card = (tmp / "reports" / "DATASET_CARD.md").read_text()
    assert ds.attrs["config_hash"] in card
    assert ds.attrs["git_commit"] in card
    assert "2 trajectories" in card and "16 x 16 grid" in card
    assert "**Passed**" in card


def test_a_failed_validation_stops_the_pipeline(run, tmp_path):
    tmp, _ = run
    ds = open_dataset(tmp / "flow.zarr", mode="r+")
    ds.written[1] = False
    params = tmp / "params.yaml"
    try:
        code = main(["validate", "--params", str(params), "--data", str(tmp / "flow.zarr"),
                     "--report", str(tmp_path / "v.json")])  # fmt: skip
    finally:
        ds.written[1] = True
    assert code == 1
