# gpu-spectral-stack

FFT kernels written from scratch in CUDA, a GPU pseudo-spectral physics solver built on them
that generates versioned training data, and a custom PyTorch spectral-convolution layer for a
Fourier Neural Operator.

Builds on [gpu-rf-pipeline](https://github.com/Jeanm2005/gpu-rf-pipeline), which provides the
radio signal-processing side. Status and plan: [docs/ROADMAP.md](docs/ROADMAP.md).

## Build

    cmake -S . -B build
    cmake --build build -j
    ./build/spectral_smoke

Requires Linux or WSL2, an NVIDIA GPU, CUDA Toolkit 12.8 or newer, CMake 3.24 or newer, and a
C++17 compiler.