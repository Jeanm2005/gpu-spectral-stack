# gpu-spectral-stack

FFT kernels written from scratch in CUDA, a GPU pseudo-spectral physics solver built on them
that generates versioned training data, and a custom PyTorch spectral-convolution layer for a
Fourier Neural Operator.

Builds on [gpu-rf-pipeline](https://github.com/Jeanm2005/gpu-rf-pipeline), which provides the
radio signal-processing side. Status and plan: [docs/ROADMAP.md](docs/ROADMAP.md).

## Requirements

Linux or WSL2, an NVIDIA GPU, CUDA Toolkit 13.x, CMake 3.24 or newer, a C++17 compiler, and
Python 3.10 or newer with its development headers.

## Build

C++ library and tools:

```bash
cmake -S . -B build
cmake --build build -j
./build/spectral_smoke
```

Python package (PyTorch extension):

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
pip install -e . --no-build-isolation
pytest -q -rs
```