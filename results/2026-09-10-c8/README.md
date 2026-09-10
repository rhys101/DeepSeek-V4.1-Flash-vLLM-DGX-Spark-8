# C8 concurrency benchmark

C8 completed **nine batches / 72 measured requests** on the existing vision-enabled eight-Spark server, using the unchanged community benchmark. No serving configuration or model restart was needed. Every node remained healthy, with zero restarts and no OOM kills.

The eight-category mean aggregate rate changed from **170.81 to 187.80 tok/s (+9.9%)** from C6 to C8. Mean per-stream decode changed from **33.78 to 27.53 tok/s (-18.5%)**.

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

Rates are tokens/s except TTFT. Aggregate includes request startup time; decode per stream excludes the time before the first token. The headline is the unweighted mean of eight separate category batches and excludes counting. C8 was measured separately after C1–C6, with the original three C1 warmups. Prefix caching remained enabled, and C8 request tags differ from the earlier C1–C6 tags. These are single-batch measurements; output lengths can vary, and no repeated-run confidence interval is available.

The same image, 300,000-token cap, 8,192 prefill batch, eight sequence slots, DSpark k=5, target/draft graphs, FlashInfer 0.7.0rc1 and vision settings were retained. The prompts were text only. The separate cold-prefill sweep was not repeated for this concurrency extension. Tony’s pinned boot 10 reference stops at C6, so there is no measured C8 reference cell.

[Raw C8 results](bench-spark8.json) · [C1–C6 and prefill](../2026-09-10/bench-spark8.json)
