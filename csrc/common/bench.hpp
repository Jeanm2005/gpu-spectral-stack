#pragma once

// Timing harness. Every GPU operation hands its three phases to Bench::run():
//
//   upload    host-to-device copies of the inputs
//   compute   kernel launches or cuFFT execution (asynchronous: they only enqueue work)
//   download  device-to-host copy of the result
//
// run() executes the three phases `warmup` times without recording, then `runs` times with
// CUDA events around each phase. Events are timestamped on the GPU when the work queued ahead
// of them finishes, so the compute time is the kernel's own time, not the time the host took
// to launch it. One-time setup (allocation, cuFFT plan creation) must happen before run() is
// called, which keeps it out of every number.

#include <functional>

struct BenchConfig {
    int warmup = 0;  // untimed passes before measuring
    int runs = 1;    // timed passes
};

struct PhaseTime {
    double mean_ms = 0.0;
    double std_ms = 0.0;  // sample standard deviation (n - 1); zero for a single run
};

struct BenchResult {
    PhaseTime h2d;
    PhaseTime kernel;
    PhaseTime d2h;
};

class Bench {
public:
    explicit Bench(BenchConfig config) : config_(config) {}

    void run(const std::function<void()>& upload, const std::function<void()>& compute,
             const std::function<void()>& download);

    const BenchConfig& config() const { return config_; }
    const BenchResult& result() const { return result_; }

private:
    BenchConfig config_;
    BenchResult result_;
};
