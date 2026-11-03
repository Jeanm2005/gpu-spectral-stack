#include "bench.hpp"

#include "check.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace {

// A CUDA event, destroyed when it goes out of scope.
class Event {
public:
    Event() { CUDA_CHECK(cudaEventCreate(&event_)); }
    ~Event()
    {
        const cudaError_t err = cudaEventDestroy(event_);
        if (err != cudaSuccess) {
            std::fprintf(stderr, "warning: cudaEventDestroy failed: %s\n",
                         cudaGetErrorString(err));
        }
    }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;

    // Places the event in `stream`, behind everything already queued there.
    void record(cudaStream_t stream) { CUDA_CHECK(cudaEventRecord(event_, stream)); }

    // Blocks until the GPU reaches the event. Reports any error from the work before it.
    void wait() { CUDA_CHECK(cudaEventSynchronize(event_)); }

    // Milliseconds from `start` to this event, measured by the GPU.
    double ms_since(const Event& start) const
    {
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start.event_, event_));
        return ms;
    }

private:
    cudaEvent_t event_{};
};

}  // namespace

Timing time_kernel(const LaunchFn& launch, cudaStream_t stream, const TimingConfig& config)
{
    if (config.warmup < 0 || config.rounds < 1 || config.iters < 1) {
        throw std::invalid_argument("time_kernel: need warmup >= 0, rounds >= 1, iters >= 1");
    }

    for (int i = 0; i < config.warmup; ++i) {
        launch(stream);
    }
    CUDA_CHECK(cudaGetLastError());

    Event start;
    Event stop;
    std::vector<double> per_launch;
    per_launch.reserve(static_cast<std::size_t>(config.rounds));
    for (int r = 0; r < config.rounds; ++r) {
        start.record(stream);
        for (int i = 0; i < config.iters; ++i) {
            launch(stream);
        }
        stop.record(stream);
        stop.wait();
        CUDA_CHECK(cudaGetLastError());
        per_launch.push_back(stop.ms_since(start) / config.iters);
    }

    std::sort(per_launch.begin(), per_launch.end());
    const std::size_t n = per_launch.size();
    Timing t;
    t.min_ms = per_launch.front();
    t.max_ms = per_launch.back();
    t.median_ms = n % 2 == 1 ? per_launch[n / 2] : 0.5 * (per_launch[n / 2 - 1] + per_launch[n / 2]);
    return t;
}

double gflops_per_s(const Work& work, double ms)
{
    return ms > 0.0 ? work.flops / (ms * 1e6) : 0.0;  // flops / (ms * 1e-3 s) / 1e9
}

double gbytes_per_s(const Work& work, double ms)
{
    return ms > 0.0 ? work.bytes / (ms * 1e6) : 0.0;
}

Work fft_c2c_work(long long n, long long batch)
{
    const double points = static_cast<double>(n) * static_cast<double>(batch);
    Work w;
    w.flops = 5.0 * points * std::log2(static_cast<double>(n));
    w.bytes = 2.0 * 8.0 * points;  // one complex64 (8 bytes) read and one written per point
    return w;
}