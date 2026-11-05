#pragma once

// Every FFT implementation, by name: the one table shared by the PyTorch binding
// (csrc/torch/ops.cpp) and the benchmark (csrc/tools/spectral_bench.cpp). Python reads the
// names with spectral._C.fft_impls(). A new implementation is added here and nowhere else, and
// every test and benchmark picks it up.

#include "fft/fft.hpp"

#include <string>

namespace spectral::fft {

struct FftImpl {
    const char* name;
    C2cFn c2c;
    R2cFn r2c;
    C2rFn c2r;
    ColumnsFn columns;        // strided column transforms, or nullptr (2D then transposes)
    const PrunedFns* pruned;  // pruning inside the kernels, or nullptr (pruned.cu gathers)
    int max_n;                // largest complex transform
    int max_real_n;           // largest real transform
    bool real_needs_work;     // r2c and c2r need the `work` buffer (no fused kernel)
};

// In the order they were written.
inline constexpr FftImpl kFftImpls[] = {
    {"naive", &c2c_naive, &r2c_naive, &c2r_naive, nullptr, nullptr, 1 << 30, 1 << 30, true},
    {"stockham_r2", &c2c_stockham_r2, &r2c_stockham_r2, &c2r_stockham_r2, &columns_stockham_r2,
     &pruned_stockham_r2, kStockhamMaxN, 2 * kStockhamMaxN, false},
    {"stockham_r4", &c2c_stockham_r4, &r2c_stockham_r4, &c2r_stockham_r4, &columns_stockham_r4,
     &pruned_stockham_r4, kStockhamMaxN, 2 * kStockhamMaxN, false},
    {"stockham_r8", &c2c_stockham_r8, &r2c_stockham_r8, &c2r_stockham_r8, &columns_stockham_r8,
     &pruned_stockham_r8, kStockhamMaxN, 2 * kStockhamMaxN, false},
    {"stockham_r8_table", &c2c_stockham_r8_table, &r2c_stockham_r8_table,
     &c2r_stockham_r8_table, &columns_stockham_r8_table, &pruned_stockham_r8_table,
     kStockhamMaxN, 2 * kStockhamMaxN, false},
};

// The implementation called `name`, or nullptr if there is none.
inline const FftImpl* find_impl(const std::string& name)
{
    for (const FftImpl& impl : kFftImpls) {
        if (name == impl.name) {
            return &impl;
        }
    }
    return nullptr;
}

// "naive, stockham_r2, ...", for error messages.
inline std::string impl_names()
{
    std::string names;
    for (const FftImpl& impl : kFftImpls) {
        names += names.empty() ? "" : ", ";
        names += impl.name;
    }
    return names;
}

}  // namespace spectral::fft
