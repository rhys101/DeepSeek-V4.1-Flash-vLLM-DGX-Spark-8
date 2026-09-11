## ep4-matched-vllm (2026-09-11T20:35:26Z)

EP4: SGLang pinned e087e662; native H8/H16 + five verify/index files plus validated draft context/layout receipt; FlashInfer 0.6.18; TP8 EP4; native resident Engram; DSpark 5; 300K context; 8 requests; four images; 0.80 static memory, pool 3.2M; fresh restored server process. Unmodified existing vLLM comparison script and deterministic prompt tags.

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 71.32 | 79.25 | 0.201 |
| C2 | 119.17 | 70.11 | 0.294 |
| C3 | 159.45 | 62.3 | 0.405 |
| C4 | 192.27 | 54.96 | 0.23 |
| C5 | 230.94 | 53.77 | 0.274 |
| C6 | 255.66 | 49.65 | 0.283 |
| C8 | 291.33 | 42.58 | 0.319 |

### Per-stream tok/s by category

| category | C1 | C2 | C3 | C4 | C5 | C6 | C8 |
|---|---|---|---|---|---|---|---|
| coding | 111.29 | 99.64 | 89.38 | 75.23 | 78.32 | 71.32 | 60.03 |
| json | 68.05 | 66.61 | 61.24 | 50.36 | 52.55 | 50.39 | 40.6 |
| narrative | 42.46 | 35.76 | 34.9 | 30.02 | 26.76 | 23.87 | 22.09 |
| prose | 52.78 | 40.8 | 39.72 | 35.21 | 31.78 | 29.63 | 25.05 |
| math | 111.36 | 96.46 | 82.75 | 73.53 | 74.71 | 64.35 | 56.94 |
| reasoning | 82.23 | 70.69 | 60.61 | 56.83 | 52.84 | 50.04 | 44.23 |
| summary | 50.65 | 46.22 | 33.92 | 35.04 | 33.39 | 31.14 | 25.91 |
| format | 115.21 | 104.72 | 95.9 | 83.45 | 79.77 | 76.45 | 65.8 |
| ceiling_count | 129.06 | 115.7 | 111.79 | 98.47 | 93.54 | 88.44 | 77.08 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 2000 | 2950 | 1.187 | 2485.0 |
| 8000 | 11592 | 3.715 | 3120.1 |
| 32000 | 46810 | 12.606 | 3713.4 |
| 64000 | 93335 | 21.905 | 4261.0 |
