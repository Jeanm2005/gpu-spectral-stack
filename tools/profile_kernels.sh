#!/usr/bin/env bash
# Profiles kernels with Nsight Compute (ncu) and saves one report per case in docs/profiles/.
#
#   tools/profile_kernels.sh                       # the default cases below
#   tools/profile_kernels.sh fft_c2c naive 1024    # one operation, implementation, and size
#
# Each case runs spectral_bench for a single launch (no warm-up, one round, one iteration), and
# ncu records every kernel that launch runs with its full set of metric sections. Open a report in
# the Nsight Compute GUI (it runs on Windows too), or print a section in the terminal:
#
#   ncu --import docs/profiles/fft_c2c_naive_1024.ncu-rep --page details --section SpeedOfLight
#
# Under WSL2, ncu needs GPU performance counters enabled on Windows: see docs/DEVELOPMENT.md.
set -euo pipefail
cd "$(dirname "$0")/.."

bench=./build/spectral_bench
out=docs/profiles
mkdir -p "$out"

profile() {  # profile OP IMPL N
    local op=$1 impl=$2 n=$3
    local name="${op}_${impl}_${n}"
    echo "==> $name"
    ncu --set full --import-source yes --force-overwrite --export "$out/$name" \
        "$bench" --op "$op" --impl "$impl" --sizes "$n" --total 16777216 \
        --warmup 0 --rounds 1 --iters 1 > /dev/null
}

if [[ $# -eq 3 ]]; then
    profile "$1" "$2" "$3"
    exit 0
elif [[ $# -ne 0 ]]; then
    echo "usage: tools/profile_kernels.sh [OP IMPL N]" >&2
    exit 1
fi

profile fft_c2c cufft 256
profile fft_c2c cufft 4096
profile fft_c2c naive 4096
profile fft_c2c stockham_r2 4096
profile fft_c2c stockham_r4 4096
profile fft_c2c stockham_r4 256
profile scale ours 16777216