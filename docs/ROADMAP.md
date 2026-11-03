# Roadmap

Story: CUDA FFT kernels for radio signals → a GPU physics solver built on them that generates
training data → a custom PyTorch spectral-convolution layer for a Fourier Neural Operator (FNO).

Conventions
- One task = one reviewed commit (sometimes a few).
- GitHub-hosted CI has no GPU. It builds everything and runs every CPU test. GPU tests run
  locally before each commit through `tools/check_local.sh` (added in 0.3).
- Every kernel variant stays selectable, so each optimization is measured against the one
  before it.

## Phase 0: Foundations (Nov 2–15)

- [X] 0.1 Build skeleton: CMake, `spectral_core` library, toolchain smoke test, CI build job
- [X] 0.2 PyTorch extension: `spectral._C` built with torch.utils.cpp_extension, pyproject.toml,
      one smoke op callable from Python. CI: build the extension in the CUDA container
- [X] 0.3 Correctness harness: float64 NumPy reference (fft, ifft, rfft, irfft, fft2, rfft2,
      pruned variants), tolerance policy that scales with log2 N, pytest size/batch matrix,
      GPU tests skip without a GPU, `tools/check_local.sh`. CI: ruff lint + CPU tests
- [X] 0.4 Benchmark harness: Bench reports GFLOP/s (5 N log2 N convention) and GB/s,
      `spectral_bench` CLI writing JSON/CSV, Python-side timing with torch.cuda.Event,
      results in docs/results/. Local regression check against the committed CSV
- [X] 0.5 Roofline tooling: measured peak bandwidth (copy kernel) and peak FP32 (FMA kernel),
      Nsight Compute script, roofline plot, DEVELOPMENT.md trimmed to this repo

## Phase 1: FFT kernels (Nov 16 – early Jan)

- [ ] 1.1 Naive batched radix-2 Cooley–Tukey, complex-to-complex (C2C), forward and inverse
- [ ] 1.2 Stockham radix-2 in shared memory, one block per transform (N ≤ 4096)
- [ ] 1.3 Radix-4 and mixed radix 2/4/8; twiddle factors from a table vs computed on the fly
- [ ] 1.4 (optional) Multi-block FFT for N beyond shared memory (four-step algorithm)
- [ ] 1.5 Real-to-complex (R2C) and complex-to-real (C2R) through the N/2-point complex trick
- [ ] 1.6 2D FFT: row transforms then column transforms (transpose vs strided), rfft2/irfft2
- [ ] 1.7 Pruned transforms: forward that outputs only the first k modes, inverse that
      reads only k non-zero modes
- [ ] 1.8 Results: table against cuFFT by size and batch, roofline, profiling log, README
- [ ] 1.9 In gpu-rf-pipeline: `--fft-impl custom`, window functions, overlap-save FIR,
      chirp and BPSK/QPSK fixtures

CI added in this phase: build for every target architecture, CPU tests of the reference and
of the tolerance policy, Nsight reports and figures regenerated from committed data.

## Phase 2: Physics solver and data pipeline (January)

- [ ] 2.1 NumPy reference solver (float64, CPU): ground truth that also runs in CI
- [ ] 2.2 GPU pseudo-spectral 2D Navier–Stokes (vorticity form) on our rfft2: 2/3-rule
      dealiasing, Kolmogorov forcing, RK4 or IMEX time stepping. Validation: Taylor–Green
      exact decay, energy/enstrophy budget, agreement with 2.1
- [ ] 2.3 Batched generation: many trajectories per launch, Gaussian-random-field initial
      conditions, config → hash + seed recorded for every sample
- [ ] 2.4 Storage: sharded Zarr, metadata schema (resolution, dt, viscosity, seed, git commit),
      chunk-size study measured against training reads
- [ ] 2.5 Validation stage: NaN/blowup, energy drift, parameter-distribution report; a bad
      dataset fails the pipeline
- [ ] 2.6 Versioning: DVC pipeline (generate → validate → report), DVC remote, `dvc repro`
      as the one command
- [ ] 2.7 Loader: streaming Dataset, pinned memory, prefetching; GPU idle time measured with
      the PyTorch profiler
- [ ] 2.8 Throughput report: sims/s, write MB/s, loader samples/s

MLOps/CI added: a tiny pipeline config (32×32, a few trajectories, NumPy backend) runs end to
end in CI through `dvc repro`; schema checks; a dataset card generated per version.

## Phase 3: Custom neural-operator layer (February–March)

- [ ] 3.1 Training harness: YAML configs, MLflow tracking (params, metrics, artifacts, DVC
      dataset hash, git commit), seeding, checkpoints
- [ ] 3.2 Baseline FNO with neuraloperator: relative L2 error, time per epoch
- [ ] 3.3 Own SpectralConv2d on torch.fft, matching the baseline (the reference for 3.4)
- [ ] 3.4 SpectralConv2d as torch.autograd.Function on our kernels, hand-derived backward,
      gradcheck in float64, derivation written up in docs/
- [ ] 3.5 Pruned FFTs inside the layer
- [ ] 3.6 Fused truncation + per-mode complex matrix multiply
- [ ] 3.7 (optional) Tensor Cores for the complex multiply
- [ ] 3.8 End-to-end comparison against the baseline, model registry entry, eval script

MLOps/CI added: CPU tests of the layer through the torch.fft path, a one-epoch tiny training
smoke test on CPU, MLflow logging tested against a temporary file store.

## Phase 4: Write-up (first week of April)

- [ ] README and blog post: radio → physics → learned operator, gradient derivation, Nsight
      evidence, an honest section on what didn't speed up and why