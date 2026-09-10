# Memory footprint

Snapshot: 2026-09-10T18:56:23.325086+00:00. The serving profile used for the benchmark: native resident Engram, vision, 300K context and eight request slots. All eight containers were running, without an OOM flag.

Each node reports 121.69 GiB usable unified RAM. CPU and GPU allocations draw from that same pool. The NVIDIA GPU-worker counter is approximately 97.75 GiB per node. Loaded model allocation was 73.16 GiB at startup, of which approximately 23.60 GiB is the mean Engram table shard. These model figures include persistent model buffers and should not be equated with a raw checkpoint divided by eight.

The reconstructed global KV pool is 118,796 blocks × 138,240 bytes = 15.2945 GiB per node. This reproduces the exact logged capacity of 3,264,224 tokens / 10.88 × 300K contexts. TP ranks hold corresponding cache state; token capacity is not multiplied by eight. The scheduler itself allows eight concurrent requests.

| Rank | Model allocation | KV pool | GPU-worker total | Linux available RAM | Swap occupied |
|---|---:|---:|---:|---:|---:|
| 0 | 73.16 GiB | 15.29 GiB | 97.75 GiB | 5.01 GiB | 5.52 GiB |
| 1 | 73.16 GiB | 15.29 GiB | 97.75 GiB | 3.41 GiB | 3.85 GiB |
| 2 | 73.16 GiB | 15.29 GiB | 97.75 GiB | 3.83 GiB | 3.91 GiB |
| 3 | 73.16 GiB | 15.29 GiB | 97.75 GiB | 4.62 GiB | 4.16 GiB |
| 4 | 73.16 GiB | 15.29 GiB | 97.75 GiB | 4.28 GiB | 3.58 GiB |
| 5 | 73.16 GiB | 15.29 GiB | 97.75 GiB | 3.86 GiB | 3.39 GiB |
| 6 | 73.16 GiB | 15.29 GiB | 97.75 GiB | 3.53 GiB | 3.39 GiB |
| 7 | 73.16 GiB | 15.29 GiB | 97.75 GiB | 4.77 GiB | 4.27 GiB |

The model and KV columns are contained within the GPU-worker allocation; do not add them to it. Subtracting those two components leaves approximately 9.30 GiB of other GPU-worker allocations, but the live split between workspaces, reserves, graphs and communication allocations was not measured.

Linux `MemAvailable` estimates memory available for new allocations and differs from completely free pages. Swap occupancy is not extra RAM and does not establish active swapping during this snapshot. CUDA and Linux counters have different scopes and can overlap. The remaining 18.93–20.53 GiB from total minus GPU-worker usage minus `MemAvailable` is an accounting remainder, not a fully attributed OS-only category. Shared-memory counters are not added again.

At startup, free device memory after distributed initialization was 99.81–101.66 GiB. The 0.80 utilization target was 97.35 GiB; it applies to vLLM sizing, not to all system memory use. A percentage-based reserve therefore does not imply the same amount of OS-available RAM.

[Measured data](../results/2026-09-10-memory/summary.json)
