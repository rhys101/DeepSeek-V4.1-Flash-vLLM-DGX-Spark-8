# Eight Sparks compared with Tony's four Sparks

Reference: [Tony's boot 10 results at the pinned commit](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark/blob/ca662ac35193c69ace9cee37f13a94abf2eff0fc/results/boot10/bench-boot10.json). The current measurements are from 11 September 2026; [raw input files](../results/) are included.

| Measurement (tok/s) | Tony, 4 Sparks | This deployment, 8 Sparks | Ratio |
|---|---|---|---|
| Coding C1, decode per stream | 73.78 | 95.91 | 1.30× |
| Coding C4, aggregate | 123.80 | 239.60 | 1.94× |
| Coding C6, aggregate | 225.48 | 305.24 | 1.35× |
| Category mean C4, aggregate | 85.72 | 157.35 | 1.84× |
| Category mean C6, aggregate | 131.86 | 205.35 | 1.56× |

The principal settings match: 300,000-token context cap, 8,192 prefill batch tokens, eight sequence slots, memory utilization 0.80, prefix caching, DSpark k=5 (probabilistic draft sampling, block rejection, adaptive verification off), FULL_AND_PIECEWISE graphs with the same capture sizes, temperature zero and thinking off. Vision is enabled with up to four images per prompt and a 1 GiB processor cache; performance prompts are text only.

Our current stack additionally uses reduced NCCL buffers/channels and selected FlashInfer b12x dense MXFP8 kernels. Both the [build pins](build-and-pins.md) and [adaptation measurements](mia-improvements.md) matter when interpreting this comparison.

## Cold prefill

| Actual prompt tokens | Tony TTFT | This deployment TTFT | Tony prefill tok/s | This deployment prefill tok/s |
|---|---|---|---|---|
| 2,950 | 3.270 s | 1.059 s | 902.2 | 2,785.5 |
| 11,592 | 11.293 s | 4.008 s | 1,026.5 | 2,892.5 |
| 46,810 | 30.426 s | 15.844 s | 1,538.5 | 2,954.3 |
| 93,335 | 78.173 s | 33.505 s | 1,194.0 | 2,785.7 |

These are four single measurements with cold inputs and a one-token reply. Prefill rate divides actual input tokens by TTFT. No full 300K request or long-context answer-quality evaluation was performed.

## Concurrency and scaling

C6 offers faster individual streams; C8 delivers the highest measured aggregate throughput. Moving from C6 to C8 adds 15.5% mean aggregate throughput while reducing mean per-stream decode by 11.6%. These timings do not establish a universal optimum.

| Metric | C4 | C6 | C8 |
|---|---|---|---|
| Eight-category mean aggregate (tok/s) | 157.35 | 205.35 | 237.27 |
| Eight-category mean per-stream decode (tok/s) | 45.39 | 39.68 | 35.08 |
| Mean time to first token (s) | 0.290 | 0.330 | 0.383 |
| Coding aggregate (tok/s) | 239.60 | 305.24 | 341.44 |
| Coding per-stream decode (tok/s) | 67.01 | 56.14 | 48.88 |

| Category | C6 aggregate | C8 aggregate | C8 per-stream decode | C8 TTFT (s) |
|---|---|---|---|---|
| Code | 305.24 | 341.44 | 48.88 | 0.436 |
| JSON | 180.55 | 217.91 | 36.17 | 0.323 |
| Math | 284.85 | 329.08 | 46.53 | 0.316 |
| Reasoning | 197.56 | 235.38 | 32.91 | 0.303 |
| Tables | 325.81 | 370.77 | 57.40 | 0.351 |
| Summary | 115.36 | 133.20 | 20.87 | 0.853 |
| Prose | 130.29 | 142.59 | 19.90 | 0.235 |
| Narrative | 103.14 | 127.76 | 18.01 | 0.248 |
| Counting ceiling | 399.76 | 494.28 | 65.85 | 0.238 |

C8 was measured separately on the same serving configuration: nine batches, 72 measured requests, all eight nodes healthy. No C8 reference exists in the pinned Tony boot 10 results.

At C6, the category mean is 1.56× Tony's four-Spark result with twice the hardware: 25.67 versus 32.97 tok/s per Spark. That normalization describes hardware efficiency, not response latency. TP8/TP4, native resident Engram/disk staging, two selected RoCE HCAs/one HCA, NCCL settings and dense kernel routing differ. This is a deployment comparison, not an isolated test of any one difference.

## Scope

All **189 category input-token counts** and **four prefill input sizes** match Tony's reference. **86 of 189 output lengths differ**; input equality does not establish identical numerical output or quality. There is one measured batch per category/concurrency and no repeated-run confidence interval.

All eight containers remained running with zero restarts, no OOM kills and no logged model traceback. Dated base-stack clock/GPU-state checks are preserved separately; no new claim about sustained clocks or the absence of later GPU-state transitions follows from those older measurements.

[Complete tables](benchmark-tables.md) · [Methodology](benchmark-method.md) · [Validation](validation.md)
