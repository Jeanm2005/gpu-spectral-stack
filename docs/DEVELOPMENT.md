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

This laptop's GPU changes speed on its own in three ways, each of which has already corrupted a
run here. Before any benchmark that will be compared or committed:

1. **On the charger.** On battery the GPU's power limit drops to about 55 W; the core clock then
   collapses to 200-300 MHz under load and the memory clock to 9001 MHz. Plugged in, the limit
   is near 100 W. Windows power mode: Best performance.
2. **Core clock locked.** Even plugged in, the GPU hits its power limit under sustained load and
   the controller swings the core clock between about 630 and 1660 MHz once a second, so
   identical runs differ by up to 40 %. Lock it in a PowerShell started with **Run as
   administrator** (the lock is impossible from WSL, and a reboot clears it):

   ```powershell
   nvidia-smi -lgc 1200,1200     # must