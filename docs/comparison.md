# Eight Sparks compared with Tony’s four Sparks

Reference: [Tony’s boot 10 results at the pinned commit](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark/blob/ca662ac35193c69ace9cee37f13a94abf2eff0fc/results/boot10/bench-boot10.json). Both benchmark JSON files are included with the [recorded deployment](../results/2026-09-10/).

| Measurement (tok/s) | Tony, 4 Sparks | This deployment, 8 Sparks | Ratio |
|---|---:|---:|---:|
| Coding C1, decode per stream | 73.78 | 92.67 | 1.26× |
| Coding C4, aggregate | 123.80 | 241.99 | 1.95× |
| Coding C6, aggregate | 225.48 | 255.61 | 1.13× |
| Category mean C4, aggregate | 85.72 | 143.75 | 1.68× |
| Category mean C6, aggregate | 131.86 | 170.81 | 1.30× |

The principal settings match: 300,000-token context cap, 8,192 prefill batch tokens, eight sequence slots, memory utilization 0.80, prefix caching, DSpark k=5 (probabilistic/block, adaptive verification off), FULL_AND_PIECEWISE graphs with the same capture sizes, temperature zero and thinking off. Vision is enabled with up to four images per prompt and a 1 GiB processor cache; performance prompts are text only.

The build uses the reference ARM64 nightly base, pinned feature source, FlashInfer 0.7.0rc1, rebuilt stable extension and four reference attention files. The printed vLLM distribution version identifies the base wheel; the feature source has its own pin. Our model snapshot is pinned; Tony’s download script does not pin his exact snapshot. See [build and pins](build-and-pins.md).

## Cold prefill

| Actual prompt tokens | Tony, 4 Sparks TTFT | This deployment TTFT | Tony prefill tok/s | This deployment prefill tok/s |
|---:|---:|---:|---:|---:|
| 2,950 | 3.270 s | 1.156 s | 902.2 | 2,551.5 |
| 11,592 | 11.293 s | 5.727 s | 1,026.5 | 2,024.0 |
| 46,810 | 30.426 s | 17.875 s | 1,538.5 | 2,618.7 |
| 93,335 | 78.173 s | 36.381 s | 1,194.0 | 2,565.5 |

These are four single measurements with cold, unique inputs and a one-token reply. Prefill rates divide actual input tokens by TTFT.

## Concurrency and scaling

**Six concurrent requests is the practical recommendation for balancing aggregate throughput and individual-stream speed.** Moving from C6 to C8 adds 9.9% mean aggregate throughput while reducing mean per-stream decode by 18.5%. C8 is the highest measured aggregate result; these timings do not prove a universal optimum.

At C6, the mean aggregate rate is 1.30× Tony’s four-Spark result with twice the hardware: 21.35 versus 32.97 tok/s per Spark. That normalization describes hardware efficiency, not response latency. Kernel and collective timings have not been profiled sufficiently to assign the scaling gap to a particular cause.



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

## Scope of the comparison

All **189 category input-token counts** and **four prefill input sizes** match Tony’s pinned boot 10 reference. **82 of 189 output lengths differ**; input equality does not establish identical numerical output or quality. There is one measured batch per category/concurrency and no repeated-run confidence interval.

The longest measured cold prompt contains 93,335 tokens and uses a one-token reply. No full 300K request or long-context answer-quality evaluation was performed.

TP8 versus TP4, native resident Engram versus node-local disk staging, and two selected RoCE HCAs versus one are material differences. This comparison does not isolate RAM-versus-NVMe performance or establish an optimized eight-node ceiling.

All eight containers remained running with zero restarts and no OOM kills. Twenty samples per GPU showed P0, 2,171–2,190 MHz and no active throttle flags. A separate model-unloaded GPU-state probe completed on all eight GPUs without a large collapse. Neither those probes nor normal sampled clocks exclude a later state transition under the resident workload.

[Complete tables](benchmark-tables.md) · [Methodology](benchmark-method.md) · [Validation](validation.md)
