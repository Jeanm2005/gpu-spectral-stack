// Pytorch bindings for the operators in csrc/ops.
#include <Python.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>
#include "common/types.hpp"
#include "ops/scale.hpp"
#include <cstddef>

namespace {
    static_assert(sizeof(c10::complex<float>) == sizeof(cf32), "complex64 must match cf32");

    const cf32* as_cf32(const at::Tensor& t)
    {
        return reinterpret_cast<cf32*>(t.mutable_data_ptr<c10::complex<float>>());
    }

    cf32* as_cf32_mut(at::Tensor& t)
    {
        return reinterpret_cast<cf32*>(t.mutable_data_ptr<c10::complex<float>>());
    }

    void check_complex64_cuda(const at::Tensor& t, const char* name)
    {
        TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
        TORCH_CHECK(t.scalar_type() == at::kComplexFloat, name, " must be complex64, got ", t.scalar_type());
    }

    at::Tensor scale_cuda(const at::Tensor& x, double alpha)
    {
        check_complex64_cuda(x, "X");
        const c10::cuda::CUDAGuard device_guard(x.device());

        const at::Tensor xc = x.contiguous();
        at::Tensor y = at::empty_like(xc);
        spectral::scale_c2c(as_cf32(xc), as_cf32_mut(y), static_cast<std::size_t>(xc.numel()), static_cast<float>(alpha), at::cuda::getCurrentCUDAStream());
        return y;
    }
} // namespace

TORCH_LIBRARY(spectral, m)
{
    m.def("scale(Tensor x, float alpha) -> Tensor");
}

TORCH_LIBRARY_IMPL(spectral, CUDA, m)
{
    m.impl("scale", &scale_cuda);
}

extern "C" PyObject* PyInit__C(void)
{
    static PyModuleDef module_def = {
        PyModuleDef_HEAD_INIT,
        "_C",  // module name
        nullptr, // docstring
        -1, // per-interpreter state size
        nullptr, // methods
    };
    return PyModule_Create(&module_def);
}