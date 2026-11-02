"""Builds the spectral._C extension: csrc/torch bindings plus the CUDA kernels they launch."""
from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = Path(__file__).resolve().parent

SOURCES = [
    "csrc/torch/ops.cpp",
    "csrc/ops/scale.cu",
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