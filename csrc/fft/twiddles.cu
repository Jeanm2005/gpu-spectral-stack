// The twiddle-factor table read by the stockham_r8_table kernels (Task 1.3).
//
// One table of kTwiddleTableSize = 4096 factors serves every size up to 4096: the factor
// exp(-2 pi i k / span) of a span-point stage is entry k * (4096 / span). Each entry is computed
// in double precision and rounded once to float, so it is correctly rounded, where sincospif is
// accurate to about one ulp. The table is 32 KiB; the kernels read it through the read-only data
// cache.

#include "fft/stockham.cuh"

#include <cmath>
#include <mutex>
#include <vector>

namespace spectral::fft::stockham {

const cf32* twiddle_table()
{
    // One table per device, built on first use and kept until the process exits.
    static std::mutex mutex;
    static std::vector<cf32*> tables;

    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    const std::lock_guard<std::mutex> lock(mutex);
    if (tables.size() <= static_cast<std::size_t>(device)) {
        tables.resize(static_cast<std::size_t>(device) + 1, nullptr);
    }
    cf32*& table = tables[static_cast<std::size_t>(device)];
    if (table == nullptr) {
        std::vector<cf32> host(kTwiddleTableSize);
        const double pi = std::acos(-1.0);
        for (int m = 0; m < kTwiddleTableSize; ++m) {
            const double a = -2.0 * pi * m / kTwiddleTableSize;
            host[static_cast<std::size_t>(m)] = cf32{static_cast<float>(std::cos(a)),
                                                     static_cast<float>(std::sin(a))};
        }
        cf32* device_table = nullptr;
        CUDA_CHECK(cudaMalloc(&device_table, host.size() * sizeof(cf32)));
        // Synchronous: the table is complete before any kernel can read it.
        CUDA_CHECK(cudaMemcpy(device_table, host.data(), host.size() * sizeof(cf32),
                              cudaMemcpyHostToDevice));
        table = device_table;
    }
    return table;
}

}  // namespace spectral::fft::stockham
