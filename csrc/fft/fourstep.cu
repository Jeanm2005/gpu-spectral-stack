// The memory-pool setting behind the four-step workspace

#include "fft/fourstep.cuh"
#include <cstdint>
#include <mutex>
#include <vector>

namespace spectral::fft::fourstep {
    bool retain_pool_memory()
    {
        // Checked and set once per device, then remembered.
        enum class State { Unknown, Retained, Unsupported };
        static std::mutex mutex;
        static std::vector<State> states;

        int device = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        const std::lock_guard<std::mutex> lock(mutex);
        if (states.size() <= static_cast<std::size_t>(device)) {
            states.resize(static_cast<std::size_t>(device) + 1, State::Unknown);
        }
        State& state = states[static_cast<std::size_t>(device)];
        if (state == State::Unknown) {
            int supported = 0;
            CUDA_CHECK(cudaDeviceGetAttribute(&supported, cudaDevAttrMemoryPoolsSupported, device));
            if (supported == 0) {
                state = State::Unsupported;
            } else {
                cudaMemPool_t pool = nullptr;
                CUDA_CHECK(cudaDeviceGetDefaultMemPool(&pool, device));
                std::uint64_t threshold = UINT64_MAX;
                CUDA_CHECK(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold));
                state = State::Retained;
            }
        }
        return state == State::Retained;
    }
}