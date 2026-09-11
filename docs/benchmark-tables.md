# Complete benchmark tables

Generated with the unchanged upstream formatter. Its generic prompt-size description is approximate: actual category inputs in these runs span 30–303 tokens. C1–C6 and C8 are separate measurements of the same serving configuration. [Method](benchmark-method.md) and [raw evidence](../results/2026-09-11/).

## 2026-09-11-mia-a9 (2026-09-11T12:20:23Z to 2026-09-11T12:24:45Z)

2026-09-11 controlled Mia-inspired A9 test, fresh process and prefix cache. NCCL reduced buffers/channels; small-M b12x enabled only in A9. 8x DGX Spark TP8; native RAM Engram; 300000 max context, 8192 batch, 8 sequences, gmu .80, prefix cache ON; DSpark k5 probabilistic/block/adaptive false; exact FULL_AND_PIECEWISE graphs. Reference nightly base 8a728663, feature source e47aa780, FlashInfer 0.7.0rc1/07869c61, reference attention patches ca662ac3. Vision ON, image limit4, processor cache1GiB. Two selected RoCE HCAs.

Prompt set `v1`, temperature 0, thinking off, streaming; one batch per cell (C streams released together). Short prompts (about 30-120 tokens), 150-256 token budgets.

### Decode: per-stream tok/s after the first token

| category | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| coding | 95.9 | 76.3 | 72.9 | 67.0 | 59.9 | 56.1 |
| json | 57.4 | 52.6 | 53.2 | 41.3 | 38.5 | 37.7 |
| math | 83.2 | 79.3 | 66.2 | 60.6 | 59.4 | 53.6 |
| reasoning | 72.4 | 61.2 | 47.9 | 44.3 | 40.1 | 36.5 |
| format | 104.6 | 91.8 | 72.9 | 69.7 | 65.3 | 67.0 |
| summary | 47.4 | 35.6 | 30.9 | 28.4 | 26.3 | 23.8 |
| prose | 41.3 | 34.4 | 29.8 | 26.5 | 23.9 | 23.6 |
| narrative | 45.0 | 27.8 | 26.4 | 25.4 | 21.6 | 19.0 |
| counting (ceiling) | 109.5 | 100.8 | 93.6 | 80.5 | 77.8 | 71.6 |

### Aggregate throughput: tok/s across all streams (wall time, TTFT included)

| category | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| coding | 87.3 | 141.0 | 203.1 | 239.6 | 268.2 | 305.2 |
| json | 50.5 | 89.7 | 130.3 | 142.2 | 144.9 | 180.6 |
| math | 76.3 | 143.8 | 177.9 | 214.9 | 268.4 | 284.9 |
| reasoning | 67.4 | 111.4 | 129.4 | 162.7 | 185.4 | 197.6 |
| format | 82.9 | 151.0 | 179.1 | 222.2 | 260.6 | 325.8 |
| summary | 41.6 | 57.7 | 56.0 | 94.5 | 107.0 | 115.4 |
| prose | 38.7 | 63.2 | 77.9 | 93.5 | 107.6 | 130.3 |
| narrative | 42.0 | 51.1 | 72.7 | 89.1 | 100.0 | 103.1 |
| counting (ceiling) | 99.8 | 187.1 | 260.4 | 297.4 | 356.3 | 399.8 |

### TTFT: mean time to first token (s)

| category | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| coding | 0.22 | 0.23 | 0.22 | 0.25 | 0.26 | 0.26 |
| json | 0.22 | 0.27 | 0.27 | 0.27 | 0.38 | 0.28 |
| math | 0.23 | 0.25 | 0.25 | 0.25 | 0.28 | 0.28 |
| reasoning | 0.22 | 0.25 | 0.25 | 0.26 | 0.26 | 0.29 |
| format | 0.26 | 0.25 | 0.26 | 0.26 | 0.28 | 0.31 |
| summary | 0.36 | 0.43 | 2.24 | 0.55 | 0.62 | 0.70 |
| prose | 0.24 | 0.19 | 0.22 | 0.25 | 0.23 | 0.26 |
| narrative | 0.24 | 0.22 | 0.23 | 0.24 | 0.24 | 0.25 |
| counting (ceiling) | 0.22 | 0.19 | 0.21 | 0.21 | 0.22 | 0.21 |

### End-to-end per stream: tok/s over the whole request, TTFT included

| category | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| coding | 87.3 | 70.5 | 67.7 | 62.1 | 55.8 | 52.6 |
| json | 50.5 | 45.7 | 45.7 | 37.0 | 33.3 | 33.8 |
| math | 76.4 | 72.6 | 61.4 | 56.6 | 55.2 | 50.1 |
| reasoning | 67.4 | 57.1 | 45.4 | 42.0 | 38.3 | 34.8 |
| format | 82.9 | 75.6 | 62.3 | 59.8 | 55.8 | 56.1 |
| summary | 41.7 | 31.8 | 19.4 | 25.2 | 23.1 | 20.8 |
| prose | 38.7 | 32.9 | 28.5 | 25.4 | 23.1 | 22.7 |
| narrative | 42.0 | 26.8 | 25.4 | 24.6 | 20.9 | 18.5 |
| counting (ceiling) | 99.8 | 93.6 | 86.8 | 75.3 | 72.9 | 67.6 |

### Decode per stream vs boot10

| category | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| coding | 73.8 → 95.9 | 45.0 → 76.3 | 33.8 → 72.9 | 34.3 → 67.0 | 44.1 → 59.9 | 41.5 → 56.1 |
| json | 52.1 → 57.4 | 30.4 → 52.6 | 20.7 → 53.2 | 22.7 → 41.3 | 26.8 → 38.5 | 27.5 → 37.7 |
| math | 50.9 → 83.2 | 59.3 → 79.3 | 54.4 → 66.2 | 31.4 → 60.6 | 28.9 → 59.4 | 34.3 → 53.6 |
| reasoning | 37.8 → 72.4 | 42.4 → 61.2 | 36.2 → 47.9 | 26.8 → 44.3 | 20.8 → 40.1 | 24.6 → 36.5 |
| format | 55.4 → 104.6 | 59.5 → 91.8 | 41.5 → 72.9 | 35.5 → 69.7 | 53.1 → 65.3 | 35.3 → 67.0 |
| summary | 25.7 → 47.4 | 22.4 → 35.6 | 22.4 → 30.9 | 13.7 → 28.4 | 15.9 → 26.3 | 16.1 → 23.8 |
| prose | 24.4 → 41.3 | 23.6 → 34.4 | 19.4 → 29.8 | 18.4 → 26.5 | 14.8 → 23.9 | 13.8 → 23.6 |
| narrative | 24.9 → 45.0 | 16.9 → 27.8 | 16.5 → 26.4 | 14.7 → 25.4 | 11.8 → 21.6 | 9.8 → 19.0 |
| counting (ceiling) | 62.2 → 109.5 | 58.9 → 100.8 | 43.0 → 93.6 | 58.3 → 80.5 | 57.2 → 77.8 | 33.4 → 71.6 |

### Cold prefill (unique prompt, 1-token reply; TTFT = whole prefill)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 2K | 2,950 | 1.06 | 2,786 |
| 8K | 11,592 | 4.01 | 2,892 |
| 32K | 46,810 | 15.84 | 2,954 |
| 64K | 93,335 | 33.51 | 2,786 |

## 2026-09-11-mia-a9-c8 (2026-09-11T12:24:46Z to 2026-09-11T12:25:39Z)

Optimized eight-Spark profile: identical to the 2026-09-11 C1-C6 run, measured separately at eight concurrent requests. Fresh C8 request prefixes; the three C1 warmups may reuse earlier warmup prefixes and are excluded from results.

Prompt set `v1`, temperature 0, thinking off, streaming; one batch per cell (C streams released together). Short prompts (about 30-120 tokens), 150-256 token budgets.

### Decode: per-stream tok/s after the first token

| category | C8 |
|---|---|
| coding | 48.9 |
| json | 36.2 |
| math | 46.5 |
| reasoning | 32.9 |
| format | 57.4 |
| summary | 20.9 |
| prose | 19.9 |
| narrative | 18.0 |
| counting (ceiling) | 65.8 |

### Aggregate throughput: tok/s across all streams (wall time, TTFT included)

| category | C8 |
|---|---|
| coding | 341.4 |
| json | 217.9 |
| math | 329.1 |
| reasoning | 235.4 |
| format | 370.8 |
| summary | 133.2 |
| prose | 142.6 |
| narrative | 127.8 |
| counting (ceiling) | 494.3 |

### TTFT: mean time to first token (s)

| category | C8 |
|---|---|
| coding | 0.44 |
| json | 0.32 |
| math | 0.32 |
| reasoning | 0.30 |
| format | 0.35 |
| summary | 0.85 |
| prose | 0.24 |
| narrative | 0.25 |
| counting (ceiling) | 0.24 |

### End-to-end per stream: tok/s over the whole request, TTFT included

| category | C8 |
|---|---|
| coding | 44.4 |
| json | 32.0 |
| math | 43.5 |
| reasoning | 31.5 |
| format | 48.4 |
| summary | 18.1 |
| prose | 19.3 |
| narrative | 17.6 |
| counting (ceiling) | 62.0 |
