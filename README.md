# Results

Committed baselines and measured ceilings. Every figure in docs/figures/ is drawn from these
files by `python tools/make_figures.py`.

| File | Contents | Produced by |
|---|---|---|
| `bench.csv` | median time, GFLOP/s, and effective GB/s of every C++ implementation | `./build/spectral_bench --out docs/results/bench.csv` |
| `bench_torch.csv` | the same for torch.fft, to show PyTorch's overhead over raw cuFFT | `python bench/bench_torch.py --out docs/results/bench_torch.csv` |
| `peaks.json` | achievable DRAM bandwidth and FP32 throughput: the roofline's ceilings | `./build/spectral_peaks --out docs/results/peaks.json` |

## Machine and conditions

NVIDIA GeForce RTX 5080 Laptop GPU (compute capability 12.0, 60 SMs, 16 GB GDDR7), CUDA 13.4,
WSL2 on Windows. Laptop plugged in, Windows power mode "Best performance", vendor profile set
to Performance. Conditions for comparable runs are in docs/DEVELOPMENT.md.

## Measured bandwidth versus the datasheet

The measured copy bandwidth is about 520 GB/s, against 896 GB/s on the datasheet (256-bit bus
at 28 Gbps). PyTorch's own device-to-device copy reaches the same 520 GB/s on this machine, and
the number does not change with power settings, so it is what this laptop sustains under WSL2,
not a limit of the measurement kernel. All percentages of peak in this repository are relative
to the measured value, which is the honest ceiling for kernels running on this machine.