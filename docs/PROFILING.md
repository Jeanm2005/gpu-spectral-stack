# Profiling log

What Nsight Compute showed about each kernel, case by case. The reports are in docs/profiles/
(`tools/profile_kernels.sh` regenerates them; docs/DEVELOPMENT.md has the setup). Every case is a
single launch of 2^24 points, the same work as one row of docs/results/bench.csv.

Conditions: RTX 5080 Laptop GPU, CUDA 13.4, core clock locked at 1200 MHz (see
docs/results/README.md). Nsight Compute times each kernel under its own clock control and with
caches flushed between replays, so its durations run about 10 % below the benchmark's medians;
compare kernels with each other here, and with cuFFT, not with bench.csv.

Columns: time is the kernel's duration; DRAM is bytes read plus written and the rate they imply;
SM is `sm__throughput` (the busiest of the SM's pipelines, % of peak); occupancy is achieved
active warps (% of peak); bank conflicts are extra shared-memory wavefronts on loads / stores;
sectors per request is 32-byte sectors per global-load request, where 8.0 means each warp reads
256 contiguous bytes (one cf32 per thread) and anything above it means scattered reads.

## 1D complex, N = 4096 (Tasks 1.1-1.3)

| Kernel | Time (µs) | DRAM (MB, GB/s) | SM % | Occupancy % | Registers | Bank conflicts ld / st | Sectors / request |
|---|---:|---:|---:|---:|---:|---:|---:|
| scale (bandwidth reference, 2^24 points) | 416 | 246, 592 | 16 | 74 | 16 | 0 / 0 | 8.0 |
| cuFFT `vector_fft<4096>` | 476 | 238, 500 | 28 | 32 | 116 | 0.06 M / 0.09 M | 7.3 |
| naive: bit reversal + 12 radix-2 stages (13 launches) | 5845 total | 3169 total | 11-26 | 72-84 | 16-24 | 0 / 0 | 32 (bit reversal), 8-16 (stages) |
| `stockham_r2` | 557 | 247, 443 | 71 | 49 | 80 | 0.31 M / 4.56 M | 8.0 |
| `stockham_r4` | 470 | 247, 526 | 56 | 49 | 80 | 0.36 M / 4.53 M | 8.0 |
| `stockham_r8` | 473 | 248, 524 | 49 | 49 | 80 | 0.31 M / 4.49 M | 8.0 |
| `stockham_r8_table` | 500 | 249, 497 | 41 | 49 | 80 | 0.12 M / 4.79 M | 14.6 |

Findings:

- **The Stockham kernels move the minimum.** Each reads and writes its 2^24 points once: 247 MB,
  the scale kernel's traffic. From radix 4 on they take 13 % longer than that pure copy, and
  radix 8 is within 1 % of cuFFT's kernel. This is the memory-bound limit the roofline in the
  README shows.
- **Radix 2 is the one that is not memory-bound.** Twelve stages, each a barrier plus a full
  pass over shared memory, keep the SM 71 % busy and add 18 % to the time. Radix 4 halves the
  stages and radix 8 cuts them to four, after which the SM has headroom (56 %, 49 %) and the time
  stops improving: nothing is left to gain from fewer stages.
- **The twiddle table loses, and the profile says why.** It replaces `sincospif` with a load
  from a 32 KiB table, but stage s reads entries k r (4096 / span) for each thread's k: a
  stride that differs per thread, so a warp's loads land in 14.6 sectors per request instead of
  8. The SFU instructions it saves were never the bottleneck (SM 49 % with them). Net: 6 % slower
  at 4096 here, and about 28 % slower than `stockham_r8` at 1024 and 2048 in the benchmark.
- **Every Stockham kernel has about 4.5 M bank conflicts on its shared-memory stores per
  launch** (extra store wavefronts), against 0.3 M on its loads. The likely source is the stage
  store s[(j - k) R + k + r ns]: in the early stages (small ns) consecutive lanes write R
  elements apart, so a warp's stores fall into few banks; the reports count the conflicts but do
  not attribute them per stage, so this is the hypothesis to check first. They
  cost nothing visible while the kernels wait on DRAM, but they are the first thing to fix if a
  kernel stops being memory-bound (the 2D column kernel below already is not). Next step:
  pad the shared array (one cf32 every 32) or swap the store order per stage.
- **The naive kernel** reads and writes everything 13 times (3.2 GB). Its bit reversal is a
  gather at 32 sectors per request, fully uncoalesced; the stage kernels coalesce once a stage's
  half-size reaches a warp (8 sectors per request from stage 3 on).
- **Occupancy is not the limit.** At N = 4096 each block holds one transform in 32 KiB of shared
  memory, which caps occupancy at about 49 %; the 256-point radix-4 kernel runs at 96 % and is
  no faster per byte.

## 1D real to complex, N = 4096 (Task 1.5)

| Kernel | Time (µs) | DRAM (MB) | SM % | Occupancy % | Registers | Bank conflicts ld / st | Sectors / request |
|---|---:|---:|---:|---:|---:|---:|---:|
| cuFFT `vector_fft_symm_r2c<4096>` | 218 | 104 | 45 | 68 | 56 | 0.07 M / 0.61 M | 6.7 |
| `stockham_r8`, 2048-point kernel with the combine fused into its store | 261 | 110 | 76 | 79 | 48 | 0.19 M / 2.22 M | 8.0 |

- The fused kernel moves 6 % more data than cuFFT and spends 20 % more time, with the SM at 76 %
  against cuFFT's 45 %: the combine step (a `sincospif`, two shared-memory reads, and an index
  division per output) makes the real transform the first of ours to lean on the SM. In the
  benchmark, where launches overlap, the two are within 3 % (README tables). Precomputing the
  combine's twiddles per block and the store-side bank conflicts are the two levers.

## 2D real to complex, 1024 x 1024 (Task 1.6)

| Kernel | Time (µs) | DRAM (MB) | SM % | Occupancy % | Sectors / request |
|---|---:|---:|---:|---:|---:|
| cuFFT: rows `vector_fft_symm_r2c<1024>` | 220 | 104 | 43 | 44 | 6.8 |
| cuFFT: columns `regular_fft<1024>` | 316 | 116 | 16 | 29 | 8.4 |
| ours, rows (R2C, 512-point kernel) | 243 | 107 | 69 | 62 | 8.0 |
| ours, columns strided in place (`StridedIo`, 1024 points) | 495 | 118 | 38 | 64 | 20.0 |
| ours, transposed columns: transpose | 211 | 107 | 20 | 83 | 8.3 |
| ours, transposed columns: rows over the former columns (`C2cIo`) | 226 | 110 | 63 | 78 | 8.0 |
| ours, transposed columns: transpose back | 193 | 109 | 26 | 79 | 8.0 |

- **The strided column kernel moves the same bytes as a row kernel and takes twice as long.** At
  1024 points per column a block holds two columns, so a warp's load covers 2 neighbouring
  elements of 16 rows: runs of 16 bytes, 20 sectors per request against the ideal 8. DRAM traffic
  is unchanged (118 MB, the L2 merges the sectors) but the requests stall the SMs (38 %). The
  transposed method costs two extra full passes (211 + 193 µs) yet each of its kernels is
  coalesced, so it pulls ahead once columns are long: the benchmark crossover is between 1024 and
  2048 points per side.
- cuFFT's column kernel takes 316 µs to our 495 µs for the same traffic. Its sectors per request
  stay near 8 because it loads several columns per warp and transposes through shared memory as
  it goes. The equivalent for us: let StridedIo hold more columns per block than the transform
  shape needs (decouple T from the lane count), so each warp reads 32-column runs.

## Pruned 2D real to complex, 256 x 256 keeping 32 x 32 modes (Task 1.7)

| Kernel | Time (µs) | DRAM (MB) | SM % | Occupancy % |
|---|---:|---:|---:|---:|
| rows: R2C writing only m2 = 32 of 129 outputs | 177 | 73 | 67 | 80 |
| columns: strided over 32 columns, storing 2 m1 = 64 of 256 rows | 80 | 17 | 54 | 62 |

- The whole pruned transform moves 90 MB, against about 220 MB for a full rfft2 of the same
  2^24 points (the 1024 x 1024 profile above; the 67 MB real input is read once in both): the row
  pass writes a quarter of its outputs and the column
  pass runs over a quarter of the columns. 257 µs in total, 2.1x faster than cuFFT's full
  transform plus a gather in the benchmark.

## What to try next, in order of expected payoff

1. Column kernel coalescing: more columns per block in StridedIo (target 8 sectors per request).
2. Shared-memory store bank conflicts in the Stockham stages (padding or per-stage store order).
3. One-pass 2D for small arrays: below 64 x 64 a whole array fits in one block's shared memory,
   and cuFFT is 2x faster there because it reads and writes it once instead of twice.
4. Precomputed combine twiddles for the real transforms.
