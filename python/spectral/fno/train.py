"""Training an FNO on the Phase 2 flow data (Tasks 3.1-3.3).

    python -m spectral.fno.train --config configs/fno/baseline.yaml
    python -m spectral.fno.train --config configs/fno/baseline.yaml --set train.epochs=5

What a run does:

    split       whole trajectories, not windows, are held out for testing (data.test_samples):
                windows of one trajectory overlap, so splitting windows would leak test frames
                into training
    normalize   vorticity divided by its RMS over the first training trajectories; log-viscosity
                mapped to [-1, 1] over the dataset's viscosity range. Both go into the checkpoint
    train       AdamW, cosine learning-rate decay over the epochs, relative L2 loss
    evaluate    every epoch: mean relative L2 of one-step predictions over every test window
    record      parameters (the config, the dataset's config hash and DVC hash, the git commit),
                per-epoch metrics (errors, learning rate, time, throughput, peak GPU memory),
                and artifacts (the config and the best checkpoint) through the tracker
    checkpoint  <checkpoint.dir>/<run_name>/latest.pt after every epoch, best.pt whenever the
                test error improves; train.resume=true continues from latest.pt

Runs are reproducible: the initialization comes from `seed`, and the loader's window order
depends only on (seed, epoch). On the GPU, a few PyTorch kernels are nondeterministic, so
repeated runs agree closely rather than bit for bit.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import yaml

from ..flow.loader import CudaPrefetcher, WindowDataset, make_loader
from ..flow.store import git_commit, open_dataset
from .config import Config, load_config
from .model import FNO2d, relative_l2
from .tracking import Tracker, make_tracker


@dataclass
class Normalizer:
    vorticity_scale: float  # vorticity is divided by this
    log_nu_center: float  # log-viscosity maps to (log nu - center) / half_range
    log_nu_half_range: float

    @classmethod
    def fit(cls, dataset, train_samples: np.ndarray, count: int = 16) -> Normalizer:
        rows = sorted(int(i) for i in train_samples[:count])
        values = np.concatenate([np.asarray(dataset.vorticity[i]).ravel() for i in rows])
        scale = float(np.sqrt(np.mean(values.astype(np.float64) ** 2)))
        lo = math.log(dataset.config.viscosity_min)
        hi = math.log(dataset.config.viscosity_max)
        half = (hi - lo) / 2
        return cls(scale, (lo + hi) / 2, half if half > 0 else 1.0)

    def frames(self, x: torch.Tensor) -> torch.Tensor:
        return x / self.vorticity_scale

    def viscosity(self, nu: torch.Tensor) -> torch.Tensor:
        return (torch.log(nu) - self.log_nu_center) / self.log_nu_half_range


def split_samples(num_samples: int, test_samples: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """(train, test) trajectory indices: a seeded random choice of `test_samples` for testing."""
    if not 0 < test_samples < num_samples:
        raise ValueError(f"need 0 < test_samples < {num_samples}, got {test_samples}")
    order = np.random.default_rng(seed).permutation(num_samples)
    return np.sort(order[test_samples:]), np.sort(order[:test_samples])


def dvc_hash(data_path: str, lock: Path = Path("dvc.lock")) -> str:
    """The content hash dvc.lock records for `data_path`, or "unknown"."""
    if not lock.exists():
        return "unknown"
    stages = (yaml.safe_load(lock.read_text()) or {}).get("stages", {})
    for stage in stages.values():
        for out in stage.get("outs", []):
            if Path(out.get("path", "")) == Path(data_path):
                return str(out.get("md5", "unknown"))
    return "unknown"


def provenance(config: Config, dataset, device: torch.device) -> dict:
    attrs = dataset.attrs
    return {
        "dataset.config_hash": attrs.get("config_hash", "unknown"),
        "dataset.git_commit": attrs.get("git_commit", "unknown"),
        "dataset.dvc_md5": dvc_hash(config.data.path),
        "dataset.resolution": int(dataset.config.n),
        "dataset.samples": int(dataset.config.num_samples),
        "code.git_commit": git_commit(),
        "torch.version": torch.__version__,
        "device.name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
    }


def batches(loader, device: torch.device):
    """Batches on `device`: copied ahead of time on a side stream for CUDA (CudaPrefetcher)."""
    if device.type == "cuda":
        yield from CudaPrefetcher(loader, device)
    else:
        for batch in loader:
            yield {k: v.to(device) for k, v in batch.items()}


def step_inputs(batch: dict, normalizer: Normalizer):
    return normalizer.frames(batch["input"]), normalizer.viscosity(batch["viscosity"])


@torch.no_grad()
def evaluate(model, loader, normalizer: Normalizer, device: torch.device) -> float:
    """Mean relative L2 error of one-step predictions over every window in `loader`."""
    model.eval()
    total, count = 0.0, 0
    for batch in batches(loader, device):
        frames, nu = step_inputs(batch, normalizer)
        prediction = model(frames, nu)
        errors = relative_l2(prediction, normalizer.frames(batch["target"]))
        total += float(errors.sum())
        count += errors.numel()
    return total / max(count, 1)


def checkpoint_dir(config: Config) -> Path:
    return Path(config.checkpoint.dir) / config.run_name


def save_checkpoint(path: Path, state: dict) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)  # atomic: an interrupted save never leaves a broken checkpoint


def build_model(config: Config) -> FNO2d:
    m = config.model
    return FNO2d(
        config.data.inputs,
        config.data.outputs,
        modes=m.modes,
        width=m.width,
        layers=m.layers,
        projection=m.projection,
        backend=m.backend,
    )


def train(config: Config, tracker: Tracker) -> dict:
    """Trains, evaluates, checkpoints, and records one run. Returns the final metrics."""
    device = torch.device(config.train.device)
    torch.manual_seed(config.seed)

    dataset = open_dataset(config.data.path)
    if not dataset.complete:
        raise ValueError(f"{config.data.path} is incomplete")
    train_samples, test_samples = split_samples(
        dataset.config.num_samples, config.data.test_samples, config.seed
    )
    normalizer = Normalizer.fit(dataset, train_samples)

    d = config.data
    train_set = WindowDataset(
        d.path, inputs=d.inputs, outputs=d.outputs, seed=config.seed, samples=train_samples
    )
    test_set = WindowDataset(
        d.path, inputs=d.inputs, outputs=d.outputs, shuffle=False, samples=test_samples
    )
    pin = device.type == "cuda"
    bs = config.train.batch_size
    train_loader = make_loader(train_set, bs, num_workers=d.num_workers, pin_memory=pin)
    test_loader = make_loader(test_set, bs, num_workers=d.num_workers, pin_memory=pin)

    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.train.lr, weight_decay=config.train.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.train.epochs)

    ckpt_dir = checkpoint_dir(config)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    start_epoch, best = 0, math.inf
    latest = ckpt_dir / "latest.pt"
    if config.train.resume and latest.exists():
        state = torch.load(latest, map_location=device)
        if state["config"]["model"] != config.as_dict()["model"]:
            raise ValueError("the checkpoint's model config differs from this run's")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch, best = state["epoch"] + 1, state["best_test_rel_l2"]

    config_file = ckpt_dir / "config.yaml"
    config_file.write_text(yaml.safe_dump(config.as_dict(), sort_keys=False))
    params = {
        **config.flat(),
        **provenance(config, dataset, device),
        "data.train_samples": int(len(train_samples)),
        "data.train_windows": len(train_set),
        "data.test_windows": len(test_set),
        "model.parameters": sum(p.numel() for p in model.parameters()),
        "normalizer.vorticity_scale": normalizer.vorticity_scale,
        "resumed_from_epoch": start_epoch,
    }
    tracker.start(config.run_name, params)
    status = "FAILED"
    metrics: dict = {}
    try:
        tracker.log_artifact(config_file)
        for epoch in range(start_epoch, config.train.epochs):
            train_set.set_epoch(epoch)
            model.train()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            began = time.perf_counter()
            total, count = 0.0, 0
            for batch in batches(train_loader, device):
                frames, nu = step_inputs(batch, normalizer)
                errors = relative_l2(model(frames, nu), normalizer.frames(batch["target"]))
                loss = errors.mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total += float(errors.detach().sum())
                count += errors.numel()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            seconds = time.perf_counter() - began
            lr = scheduler.get_last_lr()[0]
            scheduler.step()

            test_error = evaluate(model, test_loader, normalizer, device)
            metrics = {
                "train_rel_l2": total / max(count, 1),
                "test_rel_l2": test_error,
                "lr": lr,
                "epoch_seconds": seconds,
                "train_samples_per_s": count / seconds,
            }
            if device.type == "cuda":
                metrics["peak_memory_mb"] = torch.cuda.max_memory_allocated(device) / 2**20
            tracker.log_metrics(metrics, step=epoch)
            print(
                f"epoch {epoch:4d}  train {metrics['train_rel_l2']:.4f}  "
                f"test {test_error:.4f}  {seconds:6.1f} s",
                flush=True,
            )

            best = min(best, test_error)
            state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "best_test_rel_l2": best,
                "normalizer": asdict(normalizer),
                "config": config.as_dict(),
            }
            save_checkpoint(latest, state)
            if test_error <= best:
                save_checkpoint(ckpt_dir / "best.pt", state)

        if (ckpt_dir / "best.pt").exists():
            tracker.log_artifact(ckpt_dir / "best.pt")
        status = "FINISHED"
        return {**metrics, "best_test_rel_l2": best}
    finally:
        tracker.finish(status)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, help="YAML config (default: built-in defaults)")
    parser.add_argument(
        "--set", action="append", default=[], metavar="KEY=VALUE", help="override, repeatable"
    )
    args = parser.parse_args(argv)
    config = load_config(args.config, args.set)
    t = config.tracking
    result = train(config, make_tracker(t.backend, t.experiment, t.uri))
    print(f"best test relative L2: {result['best_test_rel_l2']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
