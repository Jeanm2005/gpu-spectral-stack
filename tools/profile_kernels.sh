#!/usr/bin/env bash
# Record the Nsight Compute reports in docs/profiles/ (one launch per kernel).
#
#   tools/profile_kernels.sh [RFGPU] [BENCH_DIR]
#
# RFGPU defaults to build/rfgpu and BENCH_DIR to out/bench (python tools/gen_bench.py --out
# out/bench). Needs GPU performance counters (docs/DEVELOPMENT.md). The sections are those of
# `--set full` without PM sampling and the tensor-core charts, which add megabytes per report
# and nothing these kernels use.
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
rfgpu=${1:-$root/build/rfgpu}
bench=${2:-$root/out/bench}
profiles=$root/docs/profiles
fixtures=$root/tests/fixtures
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

sections=(ComputeWorkloadAnalysis InstructionStats LaunchStats MemoryWorkloadAnalysis
          MemoryWorkloadAnalysis_Chart MemoryWorkloadAnalysis_Tables Occupancy SchedulerStats
          SourceCounters SpeedOfLight SpeedOfLight_RooflineChart WarpStateStats
          WorkloadDistribution)
section_args=()
for section in "${sections[@]}"; do
    section_args+=(--section "$section")
done

profile() {
    local name=$1
    shift
    echo "== $name"
    ncu "${section_args[@]}" --force-overwrite -o "$profiles/$name" "$@" >/dev/null
}

for impl in naive tiled tiled_const register; do
    profile "fir_${impl}_63" "$rfgpu" fir --in "$bench/bench_tone.cf32" \
        --taps "$fixtures/lowpass.f32" --out "$tmp/y.cf32" --impl "$impl"
    profile "fir_${impl}_1025" "$rfgpu" fir --in "$bench/bench_tone.cf32" \
        --taps "$fixtures/lowpass_long.f32" --out "$tmp/y.cf32" --impl "$impl"
done
for impl in naive tiled fft; do
    profile "xcorr_$impl" "$rfgpu" xcorr --a "$bench/bench_ch0.cf32" --b "$bench/bench_ch1.cf32" \
        --out "$tmp/r.cf32" --impl "$impl"
done
profile magnitude "$rfgpu" magnitude --in "$bench/bench_tone.cf32" --out "$tmp/m.f32"
