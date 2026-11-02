#include "bench.hpp"

#include "check.hpp"

#include <cmath>
#include <cstdio>
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

    // Puts the event in the default stream, behind everything already queued there.
    void record() { CUDA_CHECK(cudaEventRecord(event_)); }
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

PhaseTime summarize(const std::vector<double>& samples)
{
    PhaseTime t;
    if (samples.empty()) {
        return t;
    }
    double sum = 0.0;
    for (double s : samples) {
        sum += s;
    }
    t.mean_ms = sum / static_cast<double>(samples.size());
    if (samples.size() > 1) {
        double sq = 0.0;
        for (double s : samples) {
            sq += (s - t.mean_ms) * (s - t.mean_ms);
        }
        t.std_ms = std::sqrt(sq / static_cast<double>(samples.size() - 1));
    }
    return t;
}

}  // namespace

void Bench::run(const std::function<void()>& upload, const std::function<void()>& compute,
                const std::function<void()>& download)
{
    for (int i = 0; i < config_.warmup; ++i) {
        upload();
        compute();
        download();
    }

    Event start, uploaded, computed, downloaded;
    std::vector<double> h2d, kernel, d2h;
    for (int i = 0; i < config_.runs; ++i) {
        start.record();
        upload();
        uploaded.record();
        compute();
        computed.record();
        download();
        downloaded.record();
        downloaded.wait();

        h2d.push_back(uploaded.ms_since(start));
        kernel.push_back(computed.ms_since(uploaded));
        d2h.push_back(downloaded.ms_since(computed));
    }
    result_.h2d = summarize(h2d);
    result_.kernel = summarize(kernel);
    result_.d2h = summarize(d2h);
}
