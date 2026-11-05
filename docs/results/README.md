# Results

Committed baselines and measured ceilings. Every figure in docs/figures/ is drawn from these
files by `python tools/make_figures.py`, and the tables in the top-level README by
`python tools/results_table.py`.

| File | Contents | Produced by |
|---|---|---|
| `bench.csv` | median time, GFLOP/s, and effective GB/s of every C++ implementation of every operation, 2^24 points per launch | `./build/spectral_bench --out docs/results/bench.csv` |
| `bench_batch.csv` | 1D complex FFTs at fixed sizes over a range of batch counts | `./build/spectral_bench --op fft_c2c --sizes 64,1024,4096 --batches 1,8,64,512,4096 --out docs/results/bench_batch.csv` |
| `bench_torch.csv` | the same for torch.fft, to show PyTorch's overhead over raw cuFFT | `python bench/bench_torch.py --out docs/results/bench_torch.csv` |
| `bench_rf.csv` | `spectral.rf` overlap-save FIR (63 and 1025 taps) and windowed spectrum on 2^22 samples, with cuFFT and with each of our FFTs | `python bench/bench_rf.py --out docs/results/bench_rf.csv` |
| `peaks.json` | achievable DRAM bandwidth and FP32 throughput: the roofline's ceilings | `./build/spectral_peaks --out docs/results/peaks.json` |

## Machine and conditions

NVIDIA GeForce RTX 5080 Laptop GPU (compute capability 12.0, 60 SMs, 16 GB GDDR7), CUDA 13.4,
WSL2 on Windows, plugged in. **Every number here was measured with the GPU core clock locked at
1200 MHz** (`nvidia-smi -lgc 1200,1200`, see docs/DEVELOPMENT.md); under the lock the clock
holds steady near 1130 MHz at about 92 W. The peaks were measured under the same lock, so every
percentage of peak compares like with like.

## Why the clock is locked

Unlocked, this laptop's GPU reaches its roughly 100 W power limit under any sustained
memory-bound load. The power controller then swings the core clock between about 630 and
1660 MHz once a second (`nvidia-smi` reports reason 0x4, software power cap, throughout; the
memory clock and the temperature stay steady). A lower core clock keeps fewer memory requests
in flight, so achieved bandwidth follows the swing: two identical benchmark runs differed by up
to 40 % per row, and neither the median nor the minimum of a row was stable, because a row
fits inside a single phase of the swing. With the clock locked, two consecutive runs agree to
within 0.5 % on nearly every row.

The cost is absolute speed: these numbers are below what the GPU reaches in short bursts at
boost clocks (a copy kernel briefly reached 757 GB/s unlocked, against the datasheet's 896).
Comparisons between implementations, which are the point of this repository, are unaffected.
