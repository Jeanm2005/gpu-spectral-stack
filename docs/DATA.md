# Datasets

How the Phase 2 training data is generated, stored, and checked. Code: `python/spectral/flow/`.

## The pipeline: one command

```bash
dvc repro          # generate -> validate -> report, for whatever params.yaml changed
dvc push           # send the dataset to the DVC remote
dvc pull           # on another machine: fetch the version dvc.lock names
```

`dvc.yaml` has three stages, all run by `python -m spectral.flow.pipeline`:

| Stage | Reads | Writes |
|---|---|---|
| generate | `params.yaml` (`dataset`, `layout`) | `data/flow.zarr` |
| validate | the dataset, `params.yaml` (`validation`) | `reports/validation.json`; exits 1 if any check fails, which stops `dvc repro` |
| report | the dataset, the validation report | `reports/DATASET_CARD.md`, `reports/samples.png` |

`dvc.lock` (committed) records the content hash of every output: a commit of the repository
pins a dataset version exactly. The dataset itself lives in the DVC cache and the remote, never
in git. The dataset card (reports/DATASET_CARD.md) is rewritten for every version: contents,
provenance (config hash, git commit, seeds), statistics, and the validation verdict.

The default remote is a local directory, set up per machine (not committed, since the path is
machine-specific):

```bash
mkdir -p ~/dvc-storage/gpu-spectral-stack
dvc remote add --local -d local ~/dvc-storage/gpu-spectral-stack
```

CI runs the same pipeline on `configs/ci-params.yaml`: four 32 x 32 trajectories through the
float64 NumPy solver, so a change that breaks generation, validation, or the card fails CI.

## What a dataset holds

Trajectories of 2D incompressible flow in vorticity form on the periodic box [0, 2π)², with
Kolmogorov forcing and linear drag:

    ∂ω/∂t + u·∇ω = ν∇²ω − αω − A k_f cos(k_f y),   u = (∂ψ/∂y, −∂ψ/∂x),   ∇²ψ = −ω

simulated by the pseudo-spectral solver (`flow.gpu`, checked against the float64 `flow.reference`;
`tests/test_flow_*.py` has the exact solutions and budgets they are validated on). Each
trajectory starts from a Gaussian random field with spectrum (|k|² + τ²)^(−α/2), has its own
viscosity drawn log-uniformly from a range, and is recorded every `record_every` solver steps.

A `DatasetConfig` (`flow.generate`) determines the contents completely. Its hash
(`config_hash`: SHA-256 of the canonical JSON of every field except `batch_size`, plus the
schema version) names the dataset. Sample *i* draws its viscosity and initial condition from its
own seed, derived from (config seed, *i*), so it does not depend on the batch size or on any
other sample, and can be regenerated alone.

## Layout on disk

A Zarr v3 group:

| Array | Shape | Type | |
|---|---|---|---|
| `vorticity` | (samples, frames, n, n) | float32 | dimensions named sample, frame, y, x |
| `viscosity` | (samples,) | float64 | ν of each trajectory |
| `seed` | (samples,) | uint32 | seed of each trajectory |
| `written` | (samples,) | bool | which samples are on disk |
| `time` | (frames,) | float64 | simulation time of each frame |

`vorticity` is sharded: one file per 16 trajectories, holding chunks of (1, 8, n, n), so a read
fetches only the chunks it needs while the file count stays at samples / 16. `written` lets an
interrupted generation resume (`write_dataset` fills in only the missing samples of a dataset
with the same hash) and lets validation refuse an incomplete dataset.

## Metadata schema

Group attributes, all required, checked by `flow.store.schema_errors` whenever a dataset is
opened (types, the schema version, that `config_hash` matches `config`, and that the summary
fields agree with it):

| Key | Type | Meaning |
|---|---|---|
| `schema_version` | int | version of this layout and of the config fields |
| `config` | object | the `DatasetConfig` that produced the data |
| `config_hash` | str | SHA-256 of the config's identity |
| `git_commit` | str | commit of the generating code, `-dirty` if it had local changes |
| `created_utc` | str | ISO 8601 creation time |
| `generator` | str | module that wrote the data |
| `equation`, `domain` | str | the PDE and the domain, in words |
| `resolution` | int | grid points per side |
| `dt`, `record_every` | number, int | time step and steps between frames |
| `viscosity_range` | [min, max] | the log-uniform viscosity range |
| `samples`, `frames` | int | dataset size |
| `solver` | object | backend, fft_impl, pruned |

## Reading the data for training

`flow.loader` streams training items from a dataset:

```python
from spectral.flow.loader import CudaPrefetcher, WindowDataset, make_loader

dataset = WindowDataset("data/flow.zarr", inputs=10, outputs=1, seed=0)
loader = make_loader(dataset, batch_size=32, num_workers=4)  # pinned memory, prefetching
for epoch in range(epochs):
    dataset.set_epoch(epoch)
    for batch in CudaPrefetcher(loader):  # next batch copied to the GPU on a side stream
        batch["input"], batch["target"], batch["viscosity"]
```

An item is a window of one trajectory: `inputs` consecutive frames and the `outputs` frames
after them. Every window appears exactly once per epoch, in an order fixed by (seed, epoch).
Each worker reads whole shards (16 trajectories in one request, the fastest pattern in the
study below) and mixes the windows of several shards in a shuffle buffer. `samples=` restricts
a dataset to some trajectories, for train / test splits.

`tools/loader_profile.py` measures what the loader delivers and how much of the time a GPU
consumer spends waiting for it, with the PyTorch profiler.

### Does the GPU wait for data?

`python tools/loader_profile.py data/flow.zarr --out docs/results/loader.csv`, on the 128²
dataset: batches of 32 windows (10 frames in, 1 out), the page cache dropped before each
setting, the GPU clock locked. The consumer stands in for a training step with `work` round
trips of our rfft2 / irfft2 per batch; GPU busy and idle time come from the profiler trace.

| GPU work per batch | Workers | Pinned | Prefetcher | Windows/s | GPU busy (ms/step) | GPU idle |
|---|---:|---|---|---:|---:|---:|
| none (loader alone) | 0 | no | no | 176 | 3.8 | 98 % |
| none | 4 | yes | yes | 4524 | 4.7 | 33 % |
| none | 8 | yes | yes | 5086 | 5.2 | 16 % |
| ~2 ms (4 round trips) | 0 | no | no | 235 | 14.6 | 89 % |
| ~2 ms | 4 | no | no | 4835 | 3.8 | 41 % |
| ~2 ms | 4 | yes | yes | 4212 | 1.9 | 74 % |
| ~10.5 ms (32 round trips) | 0 | no | no | 236 | 12.7 | 91 % |
| ~10.5 ms | 2 | yes | yes | 2883 | 10.6 | **4.5 %** |
| ~10.5 ms | 4 | yes | yes | 2903 | 10.5 | **4.5 %** |
| ~10.5 ms | 4 | no | no | 2287 | 13.1 | 5.8 % |

(Every setting at every amount of work is in the CSV.)

- **Reading in the main process starves the GPU** whatever it has to do: about 230 windows/s,
  the GPU idle 90 % of the time. Worker processes are not optional.
- **The loader tops out at 4500-5100 windows/s** with 4-8 workers, about 7 ms per batch of 32.
  A training step that takes less than that per batch waits for data: at ~2 ms of GPU work the
  GPU idles 68-74 %.
- **Past that, the GPU stays busy.** At ~10.5 ms per batch the GPU idles 4.5 % with 2 or more
  workers, pinned memory, and the prefetcher; a model at 128² with a few spectral layers is
  expected to need at least that much per batch, so the loader should not be what limits
  training.
- **Pinned memory moves the copies off the GPU's critical path.** From pageable memory the
  host-to-device copy is synchronous and shows up as extra busy time (3.8 instead of 1.9 ms per
  step at ~2 ms of work); from pinned memory, with the prefetcher, it overlaps the work.

## Chunk-size study

`python tools/chunk_study.py --out docs/results/chunk_study.csv` writes one simulated dataset
(64 trajectories of 41 frames at 128², float32, 172 MB) in each layout, then reads it back after
dropping the page cache for its files, in the three ways training reads: whole trajectories,
windows of 11 frames (10 in, 1 out: one FNO training sample), and single frames. One process,
one thread, an ext4 disk inside WSL2.

| Chunk frames | Compression | Size (MB) | Write (MB/s) | Trajectories/s | Windows/s | Frames/s |
|---:|---|---:|---:|---:|---:|---:|
| 1 | none | 172.0 | 363 | 107 | 290 | 527 |
| 1 | zstd 1 | 158.9 | 183 | 48 | 214 | 446 |
| **8** | **none** | 201.3 | 572 | **199** | **338** | **559** |
| 8 | zstd 1 | 158.9 | 561 | 144 | 299 | 348 |
| 41 | none | 172.0 | 706 | 132 | 331 | 385 |
| 41 | zstd 1 | 158.8 | 893 | 75 | 147 | 165 |

Findings, and the defaults they set:

- **Chunks of 8 frames read fastest in every pattern.** One-frame chunks pay per-chunk overhead
  on every read (a whole trajectory is 41 chunk lookups); whole-trajectory chunks make a window
  or a single frame read all 41 frames. Eight frames is the middle that wins both.
- **No compression.** Turbulent vorticity in float32 is close to incompressible: zstd saves 8 %
  of the space and costs up to half the read speed.
- **Make the frame count a multiple of the chunk.** At 41 frames the sixth 8-frame chunk holds
  one real frame and seven padding frames, which is why that layout takes 201 MB instead of 172.
- These are single-thread reads; the loader of Task 2.7 reads with several workers.
