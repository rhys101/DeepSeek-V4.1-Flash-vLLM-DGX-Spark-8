# Complete benchmark tables

| Measurement (tok/s) | Tony, 4 Sparks | This deployment, 8 Sparks | Ratio |
|---|---:|---:|---:|
| Coding C1, decode per stream | 73.78 | 92.67 | 1.26× |
| Coding C4, aggregate | 123.80 | 241.99 | 1.95× |
| Coding C6, aggregate | 225.48 | 255.61 | 1.13× |
| Category mean C4, aggregate | 85.72 | 143.75 | 1.68× |
| Category mean C6, aggregate | 131.86 | 170.81 | 1.30× |

**Throughput by concurrency** (mean of the 8 categories; the counting ceiling is excluded):

| C | aggregate tok/s | per-stream decode tok/s | mean TTFT (s) | boot 10 aggregate |
|---|---|---|---|---|
| C1 | 57.25 | 68.33 | 0.402 | 37.95 |
| C2 | 85.83 | 54.51 | 0.736 | 64.30 |
| C3 | 120.71 | 48.84 | 0.443 | 78.70 |
| C4 | 143.75 | 43.41 | 0.477 | 85.72 |
| C5 | 155.25 | 36.95 | 0.492 | 114.20 |
| C6 | 170.81 | 33.78 | 0.498 | 131.86 |

**Decode: per-stream tok/s after the first token**

| category | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| **code** | 92.7 | 74.6 | 66.4 | 70.9 | 52.2 | 48.3 |
| JSON | 70.5 | 46.6 | 43.9 | 42.6 | 33.9 | 31.3 |
| math | 88.9 | 71.5 | 67.2 | 54.6 | 53.2 | 47.1 |
| reasoning | 74.7 | 56.1 | 50.9 | 41.3 | 35.3 | 32.3 |
| tables (format) | 97.5 | 87.1 | 77.3 | 64.7 | 59.3 | 52.8 |
| summary | 46.9 | 37.6 | 31.2 | 26.4 | 21.9 | 21.0 |
| prose | 41.8 | 32.9 | 26.7 | 24.7 | 20.7 | 20.1 |
| narrative | 33.7 | 29.7 | 26.9 | 22.0 | 19.0 | 17.3 |
| counting (ceiling) | 107.2 | 95.3 | 83.8 | 76.4 | 69.6 | 61.5 |

**Aggregate throughput: tok/s across all streams, TTFT included**

| category | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| **code** | 83.0 | 124.0 | 177.0 | 242.0 | 229.5 | 255.6 |
| JSON | 58.6 | 72.9 | 94.6 | 135.9 | 134.9 | 151.2 |
| math | 76.0 | 118.7 | 169.8 | 193.0 | 230.5 | 244.9 |
| reasoning | 66.3 | 97.9 | 132.0 | 143.3 | 158.3 | 173.9 |
| tables (format) | 69.5 | 125.3 | 174.4 | 192.9 | 231.0 | 241.6 |
| summary | 33.9 | 35.1 | 75.6 | 79.6 | 85.3 | 102.7 |
| prose | 38.8 | 60.9 | 72.2 | 88.5 | 90.1 | 106.2 |
| narrative | 31.9 | 51.8 | 70.1 | 74.7 | 82.3 | 90.4 |
| counting (ceiling) | 97.5 | 168.8 | 220.8 | 269.7 | 312.0 | 333.3 |

**TTFT: mean time to first token (s)**

| category | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| code | 0.26 | 0.46 | 0.32 | 0.45 | 0.48 | 0.42 |
| JSON | 0.26 | 0.44 | 0.60 | 0.40 | 0.43 | 0.46 |
| math | 0.39 | 0.44 | 0.45 | 0.42 | 0.43 | 0.46 |
| reasoning | 0.35 | 0.40 | 0.46 | 0.39 | 0.41 | 0.45 |
| tables (format) | 0.43 | 0.47 | 0.40 | 0.43 | 0.47 | 0.52 |
| summary (~300-token passage) | 0.99 | 3.02 | 0.57 | 0.85 | 0.93 | 0.77 |
| prose | 0.26 | 0.27 | 0.37 | 0.42 | 0.44 | 0.44 |
| narrative | 0.27 | 0.40 | 0.39 | 0.45 | 0.35 | 0.47 |
| counting (ceiling) | 0.23 | 0.33 | 0.38 | 0.36 | 0.41 | 0.34 |

**Prefill: cold, unique prompt, 1-token reply** (TTFT here is the whole prefill)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 2K | 2,950 | 1.16 | 2,552 |
| 8K | 11,592 | 5.73 | 2,024 |
| 32K | 46,810 | 17.88 | 2,619 |
| 64K | 93,335 | 36.38 | 2,566 |

**Decode per stream, Tony’s four Sparks → eight Sparks**

| category | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| code | 73.8 → 92.7 | 45.0 → 74.6 | 33.8 → 66.4 | 34.3 → 70.9 | 44.1 → 52.2 | 41.5 → 48.3 |
| JSON | 52.1 → 70.5 | 30.4 → 46.6 | 20.7 → 43.9 | 22.7 → 42.6 | 26.8 → 33.9 | 27.5 → 31.3 |
| math | 50.9 → 88.9 | 59.3 → 71.5 | 54.4 → 67.2 | 31.4 → 54.6 | 28.9 → 53.2 | 34.3 → 47.1 |
| reasoning | 37.8 → 74.7 | 42.4 → 56.1 | 36.2 → 50.9 | 26.8 → 41.3 | 20.8 → 35.3 | 24.6 → 32.3 |
| tables (format) | 55.4 → 97.5 | 59.5 → 87.1 | 41.5 → 77.3 | 35.5 → 64.7 | 53.1 → 59.3 | 35.3 → 52.8 |
| summary | 25.7 → 46.9 | 22.4 → 37.6 | 22.4 → 31.2 | 13.7 → 26.4 | 15.9 → 21.9 | 16.1 → 21.0 |
| prose | 24.4 → 41.8 | 23.6 → 32.9 | 19.4 → 26.7 | 18.4 → 24.7 | 14.8 → 20.7 | 13.8 → 20.1 |
| narrative | 24.9 → 33.7 | 16.9 → 29.7 | 16.5 → 26.9 | 14.7 → 22.0 | 11.8 → 19.0 | 9.8 → 17.3 |
| counting (ceiling) | 62.2 → 107.2 | 58.9 → 95.3 | 43.0 → 83.8 | 58.3 → 76.4 | 57.2 → 69.6 | 33.4 → 61.5 |

**Aggregate throughput, Tony’s four Sparks → eight Sparks**

| category | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| code | 66.5 → 83.0 | 81.1 → 124.0 | 92.4 → 177.0 | 123.8 → 242.0 | 196.7 → 229.5 | 225.5 → 255.6 |
| JSON | 43.7 → 58.6 | 49.2 → 72.9 | 51.9 → 94.6 | 76.3 → 135.9 | 94.7 → 134.9 | 130.4 → 151.2 |
| math | 45.6 → 76.0 | 105.5 → 118.7 | 148.4 → 169.8 | 110.2 → 193.0 | 127.9 → 230.5 | 182.7 → 244.9 |
| reasoning | 35.0 → 66.3 | 75.3 → 97.9 | 96.5 → 132.0 | 97.0 → 143.3 | 94.1 → 158.3 | 133.6 → 173.9 |
| tables (format) | 44.0 → 69.5 | 94.3 → 125.3 | 101.7 → 174.4 | 112.0 → 192.9 | 215.1 → 231.0 | 182.1 → 241.6 |
| summary | 21.9 → 33.9 | 37.6 → 35.1 | 38.2 → 75.6 | 43.1 → 79.6 | 64.0 → 85.3 | 73.5 → 102.7 |
| prose | 22.9 → 38.8 | 41.5 → 60.9 | 55.2 → 72.2 | 69.9 → 88.5 | 67.9 → 90.1 | 74.8 → 106.2 |
| narrative | 23.8 → 31.9 | 30.0 → 51.8 | 45.2 → 70.1 | 53.4 → 74.7 | 53.1 → 82.3 | 52.4 → 90.4 |
| counting (ceiling) | 57.3 → 97.5 | 109.0 → 168.8 | 118.6 → 220.8 | 208.8 → 269.7 | 259.9 → 312.0 | 184.9 → 333.3 |

**TTFT, Tony’s four Sparks → eight Sparks**

| category | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| code | 0.31 → 0.26 | 0.44 → 0.46 | 0.49 → 0.32 | 0.51 → 0.45 | 0.40 → 0.48 | 0.42 → 0.42 |
| JSON | 0.33 → 0.26 | 0.50 → 0.44 | 0.55 → 0.60 | 0.57 → 0.40 | 0.45 → 0.43 | 0.42 → 0.46 |
| math | 0.47 → 0.39 | 0.35 → 0.44 | 0.38 → 0.45 | 0.46 → 0.42 | 0.62 → 0.43 | 0.42 → 0.46 |
| reasoning | 0.44 → 0.35 | 0.35 → 0.40 | 0.42 → 0.46 | 0.55 → 0.39 | 0.57 → 0.41 | 0.41 → 0.45 |
| tables (format) | 0.51 → 0.43 | 0.40 → 0.47 | 0.57 → 0.40 | 0.59 → 0.43 | 0.47 → 0.47 | 0.47 → 0.52 |
| summary (~300-token passage) | 0.79 → 0.99 | 0.77 → 3.02 | 3.18 → 0.57 | 1.28 → 0.85 | 1.33 → 0.93 | 1.02 → 0.77 |
| prose | 0.38 → 0.26 | 0.39 → 0.27 | 0.31 → 0.37 | 0.32 → 0.42 | 0.36 → 0.44 | 0.48 → 0.44 |
| narrative | 0.29 → 0.27 | 0.32 → 0.40 | 0.44 → 0.39 | 0.35 → 0.45 | 0.49 → 0.35 | 0.37 → 0.47 |
| counting (ceiling) | 0.34 → 0.23 | 0.35 → 0.33 | 0.46 → 0.38 | 0.40 → 0.36 | 0.35 → 0.41 | 0.53 → 0.34 |

**Prefill: cold, unique prompt, 1-token reply** (TTFT here is the whole prefill) — boot 10 → eight Sparks

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 2K | 2,950 | 3.27 → 1.16 | 902 → 2,552 |
| 8K | 11,592 | 11.29 → 5.73 | 1,026 → 2,024 |
| 32K | 46,810 | 30.43 → 17.88 | 1,539 → 2,619 |
| 64K | 93,335 | 78.17 → 36.38 | 1,194 → 2,566 |

**End-to-end per stream: tok/s over the whole request**

| category | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| code | 83.0 | 63.9 | 60.3 | 61.4 | 46.6 | 44.0 |
| JSON | 58.6 | 37.9 | 33.9 | 35.8 | 29.2 | 27.1 |
| math | 76.1 | 62.1 | 58.7 | 49.2 | 48.0 | 42.7 |
| reasoning | 66.3 | 50.7 | 45.8 | 38.4 | 33.1 | 30.3 |
| tables (format) | 69.6 | 62.7 | 59.9 | 51.3 | 47.1 | 41.9 |
| summary | 33.9 | 18.4 | 27.0 | 22.1 | 18.6 | 18.4 |
| prose | 38.8 | 31.0 | 25.0 | 22.9 | 19.4 | 18.9 |
| narrative | 31.9 | 27.5 | 25.2 | 20.7 | 18.4 | 16.5 |
| counting (ceiling) | 97.5 | 84.4 | 74.2 | 68.8 | 62.4 | 56.8 |


All **189 category input-token counts** and **four prefill input sizes** match Tony’s pinned boot 10 reference. **82 of 189 output lengths differ**; input equality does not establish identical numerical output or quality. There is one measured batch per category/concurrency and no repeated-run confidence interval.

The longest measured cold prompt contains 93,335 tokens and uses a one-token reply. No full 300K request or long-context answer-quality evaluation was performed.

TP8 versus TP4, native resident Engram versus node-local disk staging, and two selected RoCE HCAs versus one are material differences. This comparison does not isolate RAM-versus-NVMe performance or establish an optimized eight-node ceiling.

Per-stream category cells are recomputed from request timings before rounding, matching the unchanged upstream reporter. Headline values are the original rounded benchmark output. Counting is excluded from the eight-category mean.

Raw files: [eight-Spark benchmark](../results/2026-09-10/bench-spark8.json), [Tony boot 10](../results/2026-09-10/bench-tony-boot10.json). See [method and reproduction commands](benchmark-method.md).

## C8 results


| Metric | C4 | C6 | C8 |
|---|---:|---:|---:|
| Eight-category mean aggregate (tok/s) | 143.75 | 170.81 | 187.80 |
| Eight-category mean per-stream decode (tok/s) | 43.41 | 33.78 | 27.53 |
| Mean time to first token (s) | 0.477 | 0.498 | 0.570 |
| Coding aggregate (tok/s) | 241.99 | 255.61 | 266.91 |
| Coding per-stream decode (tok/s) | 70.92 | 48.34 | 37.58 |

| Category | C6 aggregate | C8 aggregate | C8 per-stream decode | C8 TTFT (s) |
|---|---:|---:|---:|---:|
| Code | 255.61 | 266.91 | 37.58 | 0.616 |
| JSON | 151.20 | 166.72 | 25.48 | 0.570 |
| Math | 244.94 | 266.38 | 37.12 | 0.528 |
| Reasoning | 173.90 | 202.13 | 28.26 | 0.538 |
| Tables | 241.61 | 262.50 | 43.32 | 0.595 |
| Summary | 102.66 | 116.53 | 18.21 | 0.913 |
| Prose | 106.19 | 118.12 | 16.16 | 0.393 |
| Narrative | 90.40 | 103.12 | 14.10 | 0.410 |
| Counting ceiling | 333.26 | 335.35 | 50.67 | 0.958 |

C8 was measured separately on the same serving configuration: nine batches, 72 measured requests, all eight nodes healthy. No C8 reference exists in the pinned Tony boot 10 results. See [the C8 evidence](../results/2026-09-10-c8/).
