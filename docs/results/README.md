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
1200 MHz** (`nvidia-smi -lgc 1200,1200`, see the checklist below); under the lock the clock
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


## Benchmark checklist

This laptop's GPU changes speed on its own. Before any benchmark that will be compared or
committed:

1. **On the charger.** On battery the GPU's power limit drops to about 55 W; the core clock then
   collapses to 200-300 MHz under load. Plugged in, the limit is near 100 W. Windows power mode:
   Best performance.
2. **Core clock locked**, in a PowerShell started with **Run as administrator** (the lock cannot
   be set from WSL, and a reboot clears it):

   ```powershell
   nvidia-smi -lgc 1200,1200     # lock; undo with:  nvidia-smi -rgc
   ```

   Check it holds under load: the core clock should stay flat (about 1130 MHz here).

   ```bash
   ./build/spectral_bench --op fft_c2c --impl cufft --sizes 512 --warmup 0 --rounds 1000 > /dev/null &
   timeout 8 nvidia-smi --query-gpu=clocks.sm,power.draw,clocks_event_reasons.active --format=csv -lms 200
   wait
   ```

   A quicker check: `./build/spectral_peaks | grep fp32` reports about 17,000 GFLOP/s locked
   and about 34,000 unlocked. If the clock drifts, lock lower (`-lgc 1100,1100`) and re-measure
   the peaks too.
3. **Nothing else on the GPU** (games, video).

Then compare against the committed baseline:

```bash
./build/spectral_bench --out out/bench.csv
python tools/bench_compare.py docs/results/bench.csv out/bench.csv
```

Under the lock, consecutive runs agree to within 0.5 %, so a 10 % regression threshold flags only
real changes. Update the files here only in a commit whose purpose includes the performance
change, with the `bench_compare` output in the commit message. A full refresh:

```bash
./build/spectral_peaks --out docs/results/peaks.json
./build/spectral_bench --out docs/results/bench.csv
./build/spectral_bench --op fft_c2c --sizes 64,1024,4096 --batches 1,8,64,512,4096 \
    --out docs/results/bench_batch.csv
python bench/bench_torch.py --out docs/results/bench_torch.csv
python tools/make_figures.py
python tools/results_table.py
```