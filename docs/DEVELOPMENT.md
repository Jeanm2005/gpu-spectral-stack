# Development

Setup notes for building, testing, and profiling. For the quick start, see the
[README](../README.md).

## Reference environment

All results in this repository were measured on:

| Component | Version |
|---|---|
| GPU | NVIDIA GeForce RTX 5080 Laptop GPU (compute capability 12.0, 16 GB) |
| Windows driver | 617.42 (Studio) |
| OS | Ubuntu 26.04 (GCC 15, glibc 2.43) under WSL2 |
| CUDA Toolkit | 13.4, in `/usr/local/cuda-13.4`, with its `bin` on `PATH` |
| CMake | 4.2 (3.24 or newer is required) |
| Python | 3.14, with the packages in `requirements.txt` |

`cmake -S . -B build -DCMAKE_BUILD_TYPE=Release` needs no extra flags with this toolchain.

The first MVP measurements (iterations 0 to 2 in the profiling log) were taken with driver
572.76 and CUDA 12.8.1. CUDA 12.8's `nvcc` does not accept GCC 15, so that setup built with
`-DCMAKE_CXX_COMPILER=g++-14 -DCMAKE_CUDA_HOST_COMPILER=g++-14`. Every variant was re-timed on
CUDA 13.4 and matches within run-to-run noise.

## Building

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release       # rfgpu and the rfgpu_ops library
cmake --build build -j

cmake -S compiler -B compiler/build                    # sigflowc, unit and golden tests
cmake --build compiler/build -j

(cd compiler/viewer && npm ci && npm test)             # DAG viewer
```

## Testing

| Suite | Command | Needs a GPU |
|---|---|---|
| CPU reference and fixtures | `pytest -q tests/` (skips the GPU tests without one) | No |
| GPU against reference | `pytest -q tests/test_pipeline.py` | Yes |
| Compiler unit and golden tests | `ctest --test-dir compiler/build --output-on-failure` | No |
| Compiler end to end | `cd compiler && pytest -q tests/e2e` | Yes |
| Viewer | `cd compiler/viewer && npm test` | No |
| Compiler fuzzing | see "Fuzzing" in `compiler/README.md` (needs Clang) | No |

CI runs every suite that does not need a GPU.

## Profiling

- **CUDA events** (`rfgpu ... --bench 20 --warmup 3`) give kernel and transfer times. They work
  everywhere and are the source of every timing in the profiling log.
- **Nsight Systems** (`nsys profile`) records the timeline and counts kernel launches.
- **Nsight Compute** gives per-kernel metrics. `tools/profile_kernels.sh` records a report for
  every rfgpu kernel variant, and `compiler/tools/measure.py --ncu-out docs/profiles` one for
  each compiled SigFlow program. The reports in `docs/profiles/` can be opened in the Nsight
  Compute GUI or read with `ncu --import`, which needs no GPU.

### Enabling GPU performance counters under WSL2

`ncu` fails with `ERR_NVGPUCTRPERM` until the Windows driver allows non-administrator access to
the GPU performance counters. The setting is in the NVIDIA Control Panel, but hidden until
developer settings are turned on:

1. Open the NVIDIA Control Panel as an administrator.
2. In the menu bar, open **Desktop** and check **Enable Developer Settings**.
3. A **Developer** section appears in the task list on the left. Select **Manage GPU
   Performance Counters**.
4. Choose **Allow access to the GPU performance counters to all users**, then **Apply**.

The setting can reset after a driver update, so repeat it after updating.

## Figures

`tools/make_figures.py` produces every chart in `docs/figures/` from data files in
`docs/results/`, so the figures can be regenerated without re-profiling:

```bash
pip install -r tools/requirements-figures.txt
python tools/make_figures.py collect-ncu      # metrics from docs/profiles/*.ncu-repz (needs ncu)
python tools/make_figures.py collect-times    # re-time every rfgpu variant (needs a GPU)
python tools/make_figures.py plot             # docs/results -> docs/figures
```
