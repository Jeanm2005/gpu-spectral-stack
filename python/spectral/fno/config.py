"""Training configuration (Task 3.1): YAML files in configs/fno/, with command-line overrides.

    config = load_config("configs/fno/baseline.yaml", ["train.epochs=5", "model.width=16"])

Every field has a default; a YAML file sets any subset of them, and an override is
`section.field=value` with the value parsed as YAML (so 5, 1.0e-3, true, and null work). Unknown
sections or fields are errors, so a typo cannot be silently ignored.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import yaml


@dataclass
class DataConfig:
    path: str = "data/flow.zarr"
    inputs: int = 10  # frames in
    outputs: int = 1  # frames predicted
    test_samples: int = 128  # whole trajectories held out, never seen in training
    num_workers: int = 4


@dataclass
class ModelConfig:
    modes: int = 12
    width: int = 32
    layers: int = 4
    projection: int = 128
    backend: str = "torch"  # spectral convolution: see fno.model.SPECTRAL_CONV_BACKENDS


@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size: int = 32
    lr: float = 1.0e-3
    weight_decay: float = 1.0e-4
    device: str = "cuda"
    resume: bool = False  # continue from the run's latest checkpoint if there is one


@dataclass
class TrackingConfig:
    backend: str = "mlflow"  # "mlflow" or "json"
    experiment: str = "fno"
    uri: str | None = None  # MLflow tracking URI, or the JSON tracker's root (default runs/)


@dataclass
class CheckpointConfig:
    dir: str = "runs/checkpoints"  # checkpoints go to <dir>/<run_name>/


@dataclass
class Config:
    run_name: str = "fno"
    seed: int = 0
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)

    @classmethod
    def from_dict(cls, d: dict) -> Config:
        d = dict(d or {})
        sections = {f.name: f for f in fields(cls) if f.name in _SECTIONS}
        unknown = set(d) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        kwargs = {}
        for name, value in d.items():
            if name in sections:
                kwargs[name] = _section(_SECTIONS[name], value, name)
            else:
                kwargs[name] = value
        return cls(**kwargs)

    def as_dict(self) -> dict:
        return asdict(self)

    def flat(self) -> dict:
        """Every field as `section.field: value`, for logging as run parameters."""
        out = {}
        for key, value in self.as_dict().items():
            if isinstance(value, dict):
                out.update({f"{key}.{k}": v for k, v in value.items()})
            else:
                out[key] = value
        return out


_SECTIONS = {
    "data": DataConfig,
    "model": ModelConfig,
    "train": TrainConfig,
    "tracking": TrackingConfig,
    "checkpoint": CheckpointConfig,
}


def _section(cls, values: dict | None, name: str):
    values = dict(values or {})
    known = {f.name for f in fields(cls)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"unknown keys in {name}: {sorted(unknown)}")
    return cls(**values)


def apply_override(d: dict, override: str) -> None:
    """Sets `section.field=value` (or `field=value` at the top level) in a config dict."""
    key, sep, raw = override.partition("=")
    if not sep or not key:
        raise ValueError(f"an override looks like section.field=value, got {override!r}")
    value = yaml.safe_load(raw)
    parts = key.split(".")
    if len(parts) > 2:
        raise ValueError(f"overrides have at most two levels, got {key!r}")
    target = d
    if len(parts) == 2:
        target = d.setdefault(parts[0], {})
        if not isinstance(target, dict):
            raise ValueError(f"{parts[0]} is not a section")
    target[parts[-1]] = value


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    d = {}
    if path is not None:
        d = yaml.safe_load(Path(path).read_text()) or {}
    for override in overrides or []:
        apply_override(d, override)
    return Config.from_dict(d)
