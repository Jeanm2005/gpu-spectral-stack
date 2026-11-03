# Development

## Environment

- Windows with WSL2 (Ubuntu), an NVIDIA GPU, and the Windows NVIDIA driver (WSL uses it; never
  install a Linux driver inside WSL).
- CUDA Toolkit 13.x inside WSL, CMake 3.24 or newer, GCC, Python 3.10 or newer with its
  development headers (`python3.14-dev`).
- Keep the repository on the Linux filesystem (`~/...`), not under `/mnt/c`: builds there are
  many times slower.
- If the first CMake configure takes over a minute, Windows folders in `PATH` are the usual
  cause: set `appendWindowsPath = false` under `[interop]` in `/etc/wsl.conf`.

First-time setup:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
cmake -S . -B build
cmake --build build -j
pip install -e . --no-build-isolation
```

PyTorch's CUDA major version must match nvcc's (13 and 13); a minor difference only warns.

## Daily workflow

```bash
source .venv/bin/activate
# edit
tools/check_local.sh     # actionlint, ruff, C++ build, smoke tests, extension, every test
git commit ...
git push
```

`check_local.sh` runs everything CI runs plus the GPU tests CI cannot run; a commit that has not
passed it locally does not get pushed.

Rebuilding: `cmake --build build -j` after C++/CUDA changes for the command-line tools, and
`pip install -e . --no-build-isolation` after any change under `csrc/` for the Python extension.
Edits to `.py` files take effect immediately.

## Tests

- Every test that needs a GPU is marked `@pytest.mark.gpu`; `tests/conftest.py` skips those when
  no CUDA device is visible, which is how CI runs.
- Transforms are compared against the float64 references in `spectral.reference` with the
  tolerance policy in `spectral.testing` (a per-transform relative L2 error of at most
  8 eps log2 N). Use `assert_transform_close`, not elementwise comparisons.
- The `rng` fixture gives each test its own generator, seeded from the test's id.

## Benchmarks

```bash
./build/spectral_bench --out out/bench.csv
python tools/bench_compare.py docs/results/bench.csv out/bench.csv
```

Conditions for numbers that will be compared or committed: laptop plugged in, the same Windows
power mode, no other GPU work (games, video), a few minutes after start-up. WSL cannot lock GPU
clocks, which is why the regression threshold is 10 %.

`docs/results/` holds the committed baseline. Update it only in a commit whose purpose includes
the performance change, and paste the `bench_compare` output into that commit message.

## Profiling with Nsight Compute

Under WSL2, the GPU performance counters must first be enabled on Windows: NVIDIA Control Panel →
Desktop menu → Enable Developer Settings, then Developer → Manage GPU Performance Counters →
"Allow access to the GPU performance counters to all users". Without it, `ncu` reports
`ERR_NVGPUCTRPERM`.

```bash
tools/profile_kernels.sh                  # default cases
tools/profile_kernels.sh fft_c2c 1024     # one case
```

Reports go to `docs/profiles/`. `--set full` replays each kernel dozens of times to collect
every metric, so a run takes a while. Device code is built with `-lineinfo`, so the Source page
of a report maps stalls to lines of our kernels.

## Roofline and figures

```bash
./build/spectral_peaks --out docs/results/peaks.json   # measured ceilings of this GPU
python tools/make_figures.py                           # docs/figures/*.png, light and dark
```

## Continuous integration

GitHub-hosted runners have no GPU. CI runs ruff and actionlint, builds the C++/CUDA code for
compute capability 12.0 with warnings as errors, builds the PyTorch extension against the CUDA
13.0 wheels, and runs every test not marked `gpu`. Check a workflow change locally with
`actionlint` before pushing.