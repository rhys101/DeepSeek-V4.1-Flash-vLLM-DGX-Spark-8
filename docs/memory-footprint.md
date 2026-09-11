# Memory footprint

The optimized serving profile had **13.86–15.78 GiB Linux-available RAM per Spark** in the final snapshot at `2026-09-11T12:29:50.944448+00:00`. It uses native resident Engram, vision, a 300K context cap, eight request slots and memory utilization 0.80. Every container was running without OOM kills or restarts.

| Rank | Final available GiB | Minimum benchmark sample GiB | Shared memory GiB | Swap occupied GiB |
|---|---|---|---|---|
| 0 | 13.86 | 14.05 | 0.416 | 3.32 |
| 1 | 15.20 | 15.25 | 0.401 | 2.60 |
| 2 | 15.60 | 15.64 | 0.399 | 2.53 |
| 3 | 15.78 | 15.84 | 0.399 | 2.54 |
| 4 | 15.43 | 15.47 | 0.403 | 2.47 |
| 5 | 15.70 | 15.76 | 0.398 | 2.37 |
| 6 | 15.32 | 15.37 | 0.398 | 2.64 |
| 7 | 15.30 | 15.35 | 0.399 | 2.54 |

The benchmark window is `2026-09-11T12:20:23Z` through `2026-09-11T12:25:39Z`. Each rank has 158 observations in that window, sampled every two seconds. The lowest observed available memory across all eight nodes during those measurements was **14.05 GiB**. Sampling can miss short-lived peaks; this is not an enforced or guaranteed 13 GiB reserve.

Each node reports 121.69 GiB usable unified RAM. CPU and GPU allocations draw from this same pool. Model loading reports 73.16 GiB per rank. Rank 0 reports a 17.18 GiB available KV-cache budget, target/draft graph captures of 1.43/0.58 GiB, and planned cache capacity of 3,193,029 tokens (10.64 × 300K). The KV budget is a planning counter, not an independently measured allocated pool; these counters can overlap and should not be added to Linux memory totals. TP ranks hold corresponding cache state, so token capacity is not multiplied by eight. The scheduler permits eight concurrent requests.

The selected NCCL settings reduce buffers and channels. In an unloaded two-communicator/eight-rank probe, mean Linux shared memory fell from 11.61 GiB with the original settings to 0.425 GiB with the selected settings. In full serving, shared memory is 0.398–0.416 GiB in this snapshot. The probe measures system `Shmem`, not a direct allocator count of NCCL-only bytes; the controlled configuration change and serving observations support the attribution.

`MemAvailable` estimates memory available for new allocations and differs from completely free pages. Swap occupancy is not extra RAM and does not prove active swapping during the snapshot. Utilization 0.80 is a vLLM sizing input and does not imply 20% of unified RAM remains available to the operating system.

[Summary](../results/2026-09-11/memory-summary.json) · [Raw two-second samples](../results/2026-09-11/memory-samples.jsonl) · [Startup and health evidence](../results/2026-09-11/health.json)
