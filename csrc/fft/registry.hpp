#pragma once

// Every FFT implementation, by name: the one table shared by the PyTorch binding
// (csrc/torch/ops.cpp) and the benchmark (csrc/tools/spectral_bench.cpp). Python reads the
// names with spectral._C.fft_impls(). A new implementation is added here and nowhere else, and
// every test and benchmark picks it up.

#include "fft/fft.hpp"

#include <string>

namespace spectral::fft {

using C2cFn = void (*)(const cf32* x, cf32* y, int n, int batch, Direction direction,
                       cudaStream_t stream);

struct C2cImpl {
    const char* name;
    C2cFn run;
    int max_n;  // largest supported transform size
};

// In the order they were written.
inline constexpr C2cImpl kC2cImpls[] = {
    {"naive", &c2c_naive, 1 << 30},
    {"stockham_r2", &c2c_stockham_r2, kStockhamMaxN},
    {"stockham_r4", &c2c_stockham_r4, kStockhamMaxN},
    {"stockham_r8", &c2c_stockham_r8, kStockhamMaxN},
    {"stockham_r8_table", &c2c_stockham_r8_table, kStockhamMaxN},
};

// The implementation called `name`, or nullptr if there is none.
inline const C2cImpl* find_c2c(const std::string& name)
{
    for (const C2cImpl& impl : kC2cImpls) {
        if (name == impl.name) {
            return &impl;
        }
    }
    return nullptr;
}

// "naive, stockham_r2, ...", for error messages.
inline std::string c2c_names()
{
    std::string names;
    for (const C2cImpl& impl : kC2cImpls) {
        names += names.empty() ? "" : ", ";
        names += impl.name;
    }
    return names;
}

}  // namespace spectral::fft