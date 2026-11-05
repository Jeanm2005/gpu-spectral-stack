// PyTorch bindings for the operators in csrc/ops and csrc/fft.
//
// Each operator is declared once with a schema in TORCH_LIBRARY and given a CUDA implementation
// in TORCH_LIBRARY_IMPL. PyTorch's dispatcher routes every torch.ops.spectral.<name> call by the
// device of its inputs: CUDA tensors reach the functions below; CPU tensors get "Could not run
// 'spectral::<name>' with arguments from the 'CPU' backend".
//
// A binding only checks its arguments, allocates the output, and launches on PyTorch's current
// stream. The kernels know nothing about PyTorch.

#include <Python.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include "common/types.hpp"
#include "fft/registry.hpp"
#include "ops/scale.hpp"

#include <climits>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <string>

namespace {

// cf32 and c10::complex<float> are both two packed floats (re, im), so a complex64 tensor's
// storage is handed to the kernels as cf32 without a copy.
static_assert(sizeof(c10::complex<float>) == sizeof(cf32), "complex64 must match cf32");

const cf32* as_cf32(const at::Tensor& t)
{
    return reinterpret_cast<const cf32*>(t.const_data_ptr<c10::complex<float>>());
}

cf32* as_cf32_mut(at::Tensor& t)
{
    return reinterpret_cast<cf32*>(t.mutable_data_ptr<c10::complex<float>>());
}

void check_complex64_cuda(const at::Tensor& t, const char* name)
{
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.scalar_type() == at::kComplexFloat, name, " must be complex64, got ",
                t.scalar_type());
}

// The schema's `float` is a Python float, which is a C++ double.
at::Tensor scale_cuda(const at::Tensor& x, double alpha)
{
    check_complex64_cuda(x, "x");
    const c10::cuda::CUDAGuard device_guard(x.device());

    const at::Tensor xc = x.contiguous();  // no copy when x is already contiguous
    at::Tensor y = at::empty_like(xc);
    spectral::scale_c2c(as_cf32(xc), as_cf32_mut(y), static_cast<std::size_t>(xc.numel()),
                        static_cast<float>(alpha), at::cuda::getCurrentCUDAStream());
    return y;
}

// Batched complex FFT along the last dimension; every leading dimension is a batch dimension.
at::Tensor fft_c2c_cuda(const at::Tensor& x, bool inverse, c10::string_view impl_name)
{
    check_complex64_cuda(x, "x");
    TORCH_CHECK(x.dim() >= 1, "fft_c2c: x needs at least one dimension");
    const std::string name(impl_name.data(), impl_name.size());
    const spectral::fft::C2cImpl* impl = spectral::fft::find_c2c(name);
    TORCH_CHECK(impl != nullptr, "unknown FFT implementation '", name,
                "' (available: ", spectral::fft::c2c_names(), ")");
    if (x.numel() == 0) {
        return at::empty_like(x, at::MemoryFormat::Contiguous);
    }

    const int64_t n = x.size(-1);
    TORCH_CHECK((n & (n - 1)) == 0, "fft_c2c: the last dimension must be a power of two, got ",
                n);
    TORCH_CHECK(n <= impl->max_n, "fft_c2c: ", name, " supports sizes up to ", impl->max_n,
                ", got ", n);
    const int64_t batch = x.numel() / n;
    TORCH_CHECK(batch <= INT_MAX, "fft_c2c: too many transforms");

    const c10::cuda::CUDAGuard device_guard(x.device());
    const at::Tensor xc = x.contiguous();
    at::Tensor y = at::empty_like(xc);
    impl->run(as_cf32(xc), as_cf32_mut(y), static_cast<int>(n), static_cast<int>(batch),
              inverse ? spectral::fft::Direction::Inverse : spectral::fft::Direction::Forward,
              at::cuda::getCurrentCUDAStream());
    return y;
}

// spectral._C.fft_impls(): the names in fft/registry.hpp, in order, as a tuple of str.
PyObject* py_fft_impls(PyObject*, PyObject*)
{
    const auto count = static_cast<Py_ssize_t>(std::size(spectral::fft::kC2cImpls));
    PyObject* names = PyTuple_New(count);
    if (names == nullptr) {
        return nullptr;
    }
    for (Py_ssize_t i = 0; i < count; ++i) {
        PyObject* name = PyUnicode_FromString(spectral::fft::kC2cImpls[i].name);
        if (name == nullptr) {
            Py_DECREF(names);
            return nullptr;
        }
        PyTuple_SET_ITEM(names, i, name);  // the tuple takes over this reference
    }
    return names;
}

PyMethodDef module_methods[] = {
    {"fft_impls", py_fft_impls, METH_NOARGS, "Names of the FFT implementations, in order."},
    {nullptr, nullptr, 0, nullptr},  // end of the table
};

}  // namespace

TORCH_LIBRARY(spectral, m)
{
    m.def("scale(Tensor x, float alpha) -> Tensor");
    m.def("fft_c2c(Tensor x, bool inverse=False, str impl=\"naive\") -> Tensor");
}

TORCH_LIBRARY_IMPL(spectral, CUDA, m)
{
    m.impl("scale", &scale_cuda);
    m.impl("fft_c2c", &fft_c2c_cuda);
}

// Importing spectral._C from Python runs this. Loading the shared library runs the
// TORCH_LIBRARY blocks above, which register the operators under torch.ops.spectral; the module
// itself holds the plain Python functions in module_methods.
PyMODINIT_FUNC PyInit__C(void)
{
    static PyModuleDef module_def = {
        PyModuleDef_HEAD_INIT,
        "_C",            // module name
        nullptr,         // docstring
        -1,              // per-interpreter state size
        module_methods,  // methods
    };
    return PyModule_Create(&module_def);
}