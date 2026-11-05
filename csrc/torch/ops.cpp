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
#include "fft/fft2d.hpp"
#include "fft/pruned.hpp"
#include "fft/registry.hpp"
#include "ops/scale.hpp"

#include <climits>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <string>
#include <vector>

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

const spectral::fft::FftImpl& find_impl(c10::string_view impl_name)
{
    const std::string name(impl_name.data(), impl_name.size());
    const spectral::fft::FftImpl* impl = spectral::fft::find_impl(name);
    TORCH_CHECK(impl != nullptr, "unknown FFT implementation '", name,
                "' (available: ", spectral::fft::impl_names(), ")");
    return *impl;
}

// The number of transforms in x, whose last dimension holds one transform's `row` values.
int batch_of(const at::Tensor& x, int64_t row, const char* op)
{
    const int64_t batch = x.numel() / row;
    TORCH_CHECK(batch <= INT_MAX, op, ": too many transforms");
    return static_cast<int>(batch);
}

// Batched complex FFT along the last dimension; every leading dimension is a batch dimension.
at::Tensor fft_c2c_cuda(const at::Tensor& x, bool inverse, c10::string_view impl_name)
{
    check_complex64_cuda(x, "x");
    TORCH_CHECK(x.dim() >= 1, "fft_c2c: x needs at least one dimension");
    const spectral::fft::FftImpl& impl = find_impl(impl_name);
    if (x.numel() == 0) {
        return at::empty_like(x, at::MemoryFormat::Contiguous);
    }

    const int64_t n = x.size(-1);
    TORCH_CHECK((n & (n - 1)) == 0, "fft_c2c: the last dimension must be a power of two, got ",
                n);
    TORCH_CHECK(n <= impl.max_n, "fft_c2c: ", impl.name, " supports sizes up to ", impl.max_n,
                ", got ", n);
    const int batch = batch_of(x, n, "fft_c2c");

    const c10::cuda::CUDAGuard device_guard(x.device());
    const at::Tensor xc = x.contiguous();
    at::Tensor y = at::empty_like(xc);
    impl.c2c(as_cf32(xc), as_cf32_mut(y), static_cast<int>(n), batch,
             inverse ? spectral::fft::Direction::Inverse : spectral::fft::Direction::Forward,
             at::cuda::getCurrentCUDAStream());
    return y;
}

void check_real_size(int64_t n, const spectral::fft::FftImpl& impl, const char* op)
{
    TORCH_CHECK(n >= 2 && (n & (n - 1)) == 0, op,
                ": the real transform size must be a power of two of at least 2, got ", n);
    TORCH_CHECK(n <= impl.max_real_n, op, ": ", impl.name, " supports real sizes up to ",
                impl.max_real_n, ", got ", n);
}

// Scratch space for the real transforms of implementations without a fused kernel.
at::Tensor real_workspace(const spectral::fft::FftImpl& impl, int64_t n, int batch,
                          const at::Tensor& like)
{
    const int64_t count = impl.real_needs_work ? static_cast<int64_t>(batch) * (n / 2) : 0;
    return at::empty({count}, like.options().dtype(at::kComplexFloat));
}

cf32* workspace_ptr(at::Tensor& work)
{
    return work.numel() == 0 ? nullptr : as_cf32_mut(work);
}

// Batched real-to-complex FFT along the last dimension (n points in, n/2 + 1 out).
at::Tensor fft_r2c_cuda(const at::Tensor& x, c10::string_view impl_name)
{
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32, got ", x.scalar_type());
    TORCH_CHECK(x.dim() >= 1, "fft_r2c: x needs at least one dimension");
    const spectral::fft::FftImpl& impl = find_impl(impl_name);
    const int64_t n = x.size(-1);
    check_real_size(n, impl, "fft_r2c");

    std::vector<int64_t> shape = x.sizes().vec();
    shape.back() = n / 2 + 1;
    at::Tensor y = at::empty(shape, x.options().dtype(at::kComplexFloat));
    if (x.numel() == 0) {
        return y;
    }
    const int batch = batch_of(x, n, "fft_r2c");

    const c10::cuda::CUDAGuard device_guard(x.device());
    const at::Tensor xc = x.contiguous();
    at::Tensor work = real_workspace(impl, n, batch, x);
    impl.r2c(xc.const_data_ptr<float>(), as_cf32_mut(y), static_cast<int>(n), batch,
             workspace_ptr(work), at::cuda::getCurrentCUDAStream());
    return y;
}

// Batched complex-to-real FFT along the last dimension (n/2 + 1 points in, n out), the inverse
// of fft_r2c, divided by n.
at::Tensor fft_c2r_cuda(const at::Tensor& x, int64_t n, c10::string_view impl_name)
{
    check_complex64_cuda(x, "x");
    TORCH_CHECK(x.dim() >= 1, "fft_c2r: x needs at least one dimension");
    const spectral::fft::FftImpl& impl = find_impl(impl_name);
    check_real_size(n, impl, "fft_c2r");
    TORCH_CHECK(x.size(-1) == n / 2 + 1, "fft_c2r: an output of ", n,
                " points needs n/2 + 1 = ", n / 2 + 1, " inputs per transform, got ", x.size(-1));

    std::vector<int64_t> shape = x.sizes().vec();
    shape.back() = n;
    at::Tensor y = at::empty(shape, x.options().dtype(at::kFloat));
    if (x.numel() == 0) {
        return y;
    }
    const int batch = batch_of(x, n / 2 + 1, "fft_c2r");

    const c10::cuda::CUDAGuard device_guard(x.device());
    const at::Tensor xc = x.contiguous();
    at::Tensor work = real_workspace(impl, n, batch, x);
    impl.c2r(as_cf32(xc), y.mutable_data_ptr<float>(), static_cast<int>(n), batch,
             workspace_ptr(work), at::cuda::getCurrentCUDAStream());
    return y;
}

// --- 2D (Task 1.6) ------------------------------------------------------------------------------

// "auto" picks the strided column kernel when the implementation has one.
spectral::fft::Columns parse_columns(c10::string_view text, const spectral::fft::FftImpl& impl)
{
    const std::string s(text.data(), text.size());
    if (s == "auto") {
        return impl.columns != nullptr ? spectral::fft::Columns::Strided
                                       : spectral::fft::Columns::Transpose;
    }
    if (s == "strided") {
        TORCH_CHECK(impl.columns != nullptr, impl.name,
                    " has no strided column kernel; use columns=\"transpose\"");
        return spectral::fft::Columns::Strided;
    }
    TORCH_CHECK(s == "transpose", "columns must be \"auto\", \"strided\", or \"transpose\", got '",
                s, "'");
    return spectral::fft::Columns::Transpose;
}

struct Shape2d {
    int64_t n1;
    int64_t n2;
    int batch;
};

// The last two dimensions of x are one array of n1 x n2 points, stored with n2_stored values per
// row (n2 for complex or real data, n2/2 + 1 for the complex side of a real transform). The
// sizes themselves are checked by fft2d.cu.
Shape2d shape2d(const at::Tensor& x, int64_t n2, int64_t n2_stored, const char* op)
{
    TORCH_CHECK(x.dim() >= 2, op, ": x needs at least two dimensions");
    const int64_t n1 = x.size(-2);
    TORCH_CHECK(n1 >= 1 && n2 >= 1, op, ": empty transform dimensions");
    return {n1, n2, batch_of(x, n1 * n2_stored, op)};
}

at::Tensor workspace(std::size_t count, const at::Tensor& like)
{
    return at::empty({static_cast<int64_t>(count)}, like.options().dtype(at::kComplexFloat));
}

at::Tensor fft2_c2c_cuda(const at::Tensor& x, bool inverse, c10::string_view impl_name,
                         c10::string_view columns)
{
    check_complex64_cuda(x, "x");
    const spectral::fft::FftImpl& impl = find_impl(impl_name);
    const spectral::fft::Columns method = parse_columns(columns, impl);
    TORCH_CHECK(x.dim() >= 2, "fft2_c2c: x needs at least two dimensions");
    if (x.numel() == 0) {
        return at::empty_like(x, at::MemoryFormat::Contiguous);
    }
    const Shape2d s = shape2d(x, x.size(-1), x.size(-1), "fft2_c2c");

    const c10::cuda::CUDAGuard device_guard(x.device());
    const at::Tensor xc = x.contiguous();
    at::Tensor y = at::empty_like(xc);
    const auto n1 = static_cast<int>(s.n1);
    const auto n2 = static_cast<int>(s.n2);
    at::Tensor work = workspace(spectral::fft::fft2_c2c_workspace(impl, method, n1, n2, s.batch),
                                x);
    spectral::fft::fft2_c2c(
        impl, method, as_cf32(xc), as_cf32_mut(y), n1, n2, s.batch,
        inverse ? spectral::fft::Direction::Inverse : spectral::fft::Direction::Forward,
        workspace_ptr(work), at::cuda::getCurrentCUDAStream());
    return y;
}

at::Tensor fft2_r2c_cuda(const at::Tensor& x, c10::string_view impl_name,
                         c10::string_view columns)
{
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32, got ", x.scalar_type());
    const spectral::fft::FftImpl& impl = find_impl(impl_name);
    const spectral::fft::Columns method = parse_columns(columns, impl);
    TORCH_CHECK(x.dim() >= 2, "fft2_r2c: x needs at least two dimensions");
    const int64_t n2 = x.size(-1);
    check_real_size(n2, impl, "fft2_r2c");

    std::vector<int64_t> shape = x.sizes().vec();
    shape.back() = n2 / 2 + 1;
    at::Tensor y = at::empty(shape, x.options().dtype(at::kComplexFloat));
    if (x.numel() == 0) {
        return y;
    }
    const Shape2d s = shape2d(x, n2, n2, "fft2_r2c");

    const c10::cuda::CUDAGuard device_guard(x.device());
    const at::Tensor xc = x.contiguous();
    const auto n1 = static_cast<int>(s.n1);
    at::Tensor work = workspace(
        spectral::fft::fft2_r2c_workspace(impl, method, n1, static_cast<int>(n2), s.batch), x);
    spectral::fft::fft2_r2c(impl, method, xc.const_data_ptr<float>(), as_cf32_mut(y), n1,
                            static_cast<int>(n2), s.batch, workspace_ptr(work),
                            at::cuda::getCurrentCUDAStream());
    return y;
}

at::Tensor fft2_c2r_cuda(const at::Tensor& x, int64_t n2, c10::string_view impl_name,
                         c10::string_view columns)
{
    check_complex64_cuda(x, "x");
    const spectral::fft::FftImpl& impl = find_impl(impl_name);
    const spectral::fft::Columns method = parse_columns(columns, impl);
    TORCH_CHECK(x.dim() >= 2, "fft2_c2r: x needs at least two dimensions");
    check_real_size(n2, impl, "fft2_c2r");
    TORCH_CHECK(x.size(-1) == n2 / 2 + 1, "fft2_c2r: an output of ", n2,
                " columns needs n2/2 + 1 = ", n2 / 2 + 1, " input columns, got ", x.size(-1));

    std::vector<int64_t> shape = x.sizes().vec();
    shape.back() = n2;
    at::Tensor y = at::empty(shape, x.options().dtype(at::kFloat));
    if (x.numel() == 0) {
        return y;
    }
    const Shape2d s = shape2d(x, n2, n2 / 2 + 1, "fft2_c2r");

    const c10::cuda::CUDAGuard device_guard(x.device());
    const at::Tensor xc = x.contiguous();
    const auto n1 = static_cast<int>(s.n1);
    at::Tensor work = workspace(
        spectral::fft::fft2_c2r_workspace(impl, method, n1, static_cast<int>(n2), s.batch), x);
    spectral::fft::fft2_c2r(impl, method, as_cf32(xc), y.mutable_data_ptr<float>(), n1,
                            static_cast<int>(n2), s.batch, workspace_ptr(work),
                            at::cuda::getCurrentCUDAStream());
    return y;
}

// --- Pruned (Task 1.7) -------------------------------------------------------------------------

// Forward: x (..., n) -> the first `keep` outputs (..., keep). Inverse: x (..., keep) holds the
// first keep modes of n -> (..., n), the other modes taken as zero.
at::Tensor fft_c2c_pruned_cuda(const at::Tensor& x, int64_t n, int64_t keep, bool inverse,
                               c10::string_view impl_name)
{
    check_complex64_cuda(x, "x");
    TORCH_CHECK(x.dim() >= 1, "fft_c2c_pruned: x needs at least one dimension");
    const spectral::fft::FftImpl& impl = find_impl(impl_name);
    TORCH_CHECK(n >= 1 && n <= INT_MAX && keep >= 1 && keep <= n,
                "fft_c2c_pruned: need 1 <= keep <= n, got keep = ", keep, ", n = ", n);
    const int64_t row_in = inverse ? keep : n;
    TORCH_CHECK(x.size(-1) == row_in, "fft_c2c_pruned: expected ", row_in,
                " values in the last dimension, got ", x.size(-1));

    std::vector<int64_t> shape = x.sizes().vec();
    shape.back() = inverse ? n : keep;
    at::Tensor y = at::empty(shape, x.options());
    if (x.numel() == 0) {
        return y;
    }
    const int batch = batch_of(x, row_in, "fft_c2c_pruned");

    const c10::cuda::CUDAGuard device_guard(x.device());
    const at::Tensor xc = x.contiguous();
    const auto ni = static_cast<int>(n);
    const auto ki = static_cast<int>(keep);
    at::Tensor work =
        workspace(spectral::fft::fft_c2c_pruned_workspace(impl, ni, ki, batch), x);
    spectral::fft::fft_c2c_pruned(
        impl, as_cf32(xc), as_cf32_mut(y), ni, ki, batch,
        inverse ? spectral::fft::Direction::Inverse : spectral::fft::Direction::Forward,
        workspace_ptr(work), at::cuda::getCurrentCUDAStream());
    return y;
}

// x (..., n1, n2) real -> the (..., 2 m1, m2) low-frequency modes of its rfft2.
at::Tensor rfft2_modes_cuda(const at::Tensor& x, int64_t m1, int64_t m2,
                            c10::string_view impl_name)
{
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32, got ", x.scalar_type());
    TORCH_CHECK(x.dim() >= 2, "rfft2_modes: x needs at least two dimensions");
    const spectral::fft::FftImpl& impl = find_impl(impl_name);
    const int64_t n1 = x.size(-2);
    const int64_t n2 = x.size(-1);
    TORCH_CHECK(m1 >= 1 && 2 * m1 <= n1 && m2 >= 1 && m2 <= n2 / 2 + 1,
                "rfft2_modes: need 1 <= m1 <= n1/2 and 1 <= m2 <= n2/2 + 1, got m1 = ", m1,
                ", m2 = ", m2, " for ", n1, " x ", n2);

    std::vector<int64_t> shape = x.sizes().vec();
    shape[shape.size() - 2] = 2 * m1;
    shape.back() = m2;
    at::Tensor y = at::empty(shape, x.options().dtype(at::kComplexFloat));
    if (x.numel() == 0) {
        return y;
    }
    const int batch = batch_of(x, n1 * n2, "rfft2_modes");

    const c10::cuda::CUDAGuard device_guard(x.device());
    const at::Tensor xc = x.contiguous();
    const int a = static_cast<int>(n1);
    const int b = static_cast<int>(n2);
    const int p = static_cast<int>(m1);
    const int q = static_cast<int>(m2);
    at::Tensor work =
        workspace(spectral::fft::rfft2_modes_workspace(impl, a, b, p, q, batch), x);
    spectral::fft::rfft2_modes(impl, xc.const_data_ptr<float>(), as_cf32_mut(y), a, b, p, q,
                               batch, workspace_ptr(work), at::cuda::getCurrentCUDAStream());
    return y;
}

// modes (..., 2 m1, m2) -> the real (..., n1, n2) array whose rfft2 has those modes and zeros
// elsewhere.
at::Tensor irfft2_modes_cuda(const at::Tensor& modes, int64_t n1, int64_t n2,
                             c10::string_view impl_name)
{
    check_complex64_cuda(modes, "modes");
    TORCH_CHECK(modes.dim() >= 2, "irfft2_modes: modes needs at least two dimensions");
    const spectral::fft::FftImpl& impl = find_impl(impl_name);
    const int64_t two_m1 = modes.size(-2);
    const int64_t m2 = modes.size(-1);
    TORCH_CHECK(two_m1 % 2 == 0 && two_m1 >= 2 && two_m1 <= n1 && m2 >= 1 && m2 <= n2 / 2 + 1,
                "irfft2_modes: modes of shape (", two_m1, ", ", m2, ") do not fit an output of ",
                n1, " x ", n2);

    std::vector<int64_t> shape = modes.sizes().vec();
    shape[shape.size() - 2] = n1;
    shape.back() = n2;
    at::Tensor y = at::empty(shape, modes.options().dtype(at::kFloat));
    if (modes.numel() == 0) {
        return y;
    }
    const int batch = batch_of(modes, two_m1 * m2, "irfft2_modes");

    const c10::cuda::CUDAGuard device_guard(modes.device());
    const at::Tensor mc = modes.contiguous();
    const int a = static_cast<int>(n1);
    const int b = static_cast<int>(n2);
    const int p = static_cast<int>(two_m1 / 2);
    const int q = static_cast<int>(m2);
    at::Tensor work =
        workspace(spectral::fft::irfft2_modes_workspace(impl, a, b, p, q, batch), modes);
    spectral::fft::irfft2_modes(impl, as_cf32(mc), y.mutable_data_ptr<float>(), a, b, p, q,
                                batch, workspace_ptr(work), at::cuda::getCurrentCUDAStream());
    return y;
}

// spectral._C.fft_impls(): the names in fft/registry.hpp, in order, as a tuple of str.
PyObject* py_fft_impls(PyObject*, PyObject*)
{
    const auto count = static_cast<Py_ssize_t>(std::size(spectral::fft::kFftImpls));
    PyObject* names = PyTuple_New(count);
    if (names == nullptr) {
        return nullptr;
    }
    for (Py_ssize_t i = 0; i < count; ++i) {
        PyObject* name = PyUnicode_FromString(spectral::fft::kFftImpls[i].name);
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
    m.def("fft_r2c(Tensor x, str impl=\"naive\") -> Tensor");
    m.def("fft_c2r(Tensor x, int n, str impl=\"naive\") -> Tensor");
    m.def("fft2_c2c(Tensor x, bool inverse=False, str impl=\"naive\", str columns=\"auto\")"
          " -> Tensor");
    m.def("fft2_r2c(Tensor x, str impl=\"naive\", str columns=\"auto\") -> Tensor");
    m.def("fft2_c2r(Tensor x, int n2, str impl=\"naive\", str columns=\"auto\") -> Tensor");
    m.def("fft_c2c_pruned(Tensor x, int n, int keep, bool inverse=False, str impl=\"naive\")"
          " -> Tensor");
    m.def("rfft2_modes(Tensor x, int m1, int m2, str impl=\"naive\") -> Tensor");
    m.def("irfft2_modes(Tensor modes, int n1, int n2, str impl=\"naive\") -> Tensor");
}

TORCH_LIBRARY_IMPL(spectral, CUDA, m)
{
    m.impl("scale", &scale_cuda);
    m.impl("fft_c2c", &fft_c2c_cuda);
    m.impl("fft_r2c", &fft_r2c_cuda);
    m.impl("fft_c2r", &fft_c2r_cuda);
    m.impl("fft2_c2c", &fft2_c2c_cuda);
    m.impl("fft2_r2c", &fft2_r2c_cuda);
    m.impl("fft2_c2r", &fft2_c2r_cuda);
    m.impl("fft_c2c_pruned", &fft_c2c_pruned_cuda);
    m.impl("rfft2_modes", &rfft2_modes_cuda);
    m.impl("irfft2_modes", &irfft2_modes_cuda);
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