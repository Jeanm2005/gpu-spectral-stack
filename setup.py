"""Builds the spectral._C extension: csrc/torch bindings plus the CUDA kernels they launch.

Install in editable mode, without build isolation, so the build uses the PyTorch already in
the venv (an isolated build would download a different one):

    pip install -e . --no-build-isolation

Device code targets TORCH_CUDA_ARCH_LIST if it is set (CI sets "12.0"), otherwise the GPUs
visible at build time.
"""

from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = Path(__file__).resolve().parent

# Paths relative to this file, as setuptools requires. Every kernel source in CMakeLists.txt's
# spectral_core belongs here too, except the timing harness.
SOURCES = [
    "csrc/torch/ops.cpp",
    "csrc/ops/scale.cu",
    "csrc/fft/naive.cu",
    "csrc/fft/real.cu",
    "csrc/fft/stockham_r2.cu",
    "csrc/fft/stockham_r4.cu",
    "csrc/fft/stockham_r8.cu",
    "csrc/fft/stockham_r8_table.cu",
    "csrc/fft/twiddles.cu",
]

setup(
    ext_modules=[
        CUDAExtension(
            name="spectral._C",
            sources=SOURCES,
            include_dirs=[str(ROOT / "csrc")],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "-lineinfo"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
