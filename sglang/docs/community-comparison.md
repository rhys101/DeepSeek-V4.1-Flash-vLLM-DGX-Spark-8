# Matched community benchmark: EP4 versus vLLM

**111.29 tok/s single-stream coding decode**, versus **95.91** on the existing vLLM deployment (16.0% higher).

| Measurement (tok/s) | vLLM, eight Sparks | SGLang EP4, eight Sparks | EP4 change |
|---|---|---|---|
| Coding C1, decode per stream | 95.91 | 111.29 | +16.0% |
| Coding C4, aggregate | 239.60 | 274.07 | +14.4% |
| Coding C6, aggregate | 305.24 | 390.93 | +28.1% |
| Coding C8, aggregate | 341.44 | 431.82 | +26.5% |
| Category mean C4, aggregate | 157.35 | 192.27 | +22.2% |
| Category mean C6, aggregate | 205.35 | 255.66 | +24.5% |
| Category mean C8, aggregate | 237.27 | 291.33 | +22.8% |

## Workload and timing

Both use the unchanged repository `bench/v41bench.py` (SHA-256 `e0d6b2d25bd585d11fbdf39c2ddcdf7a4de8ab685af6bd42465e69f3ee6e80a8`), the same client host, eight Sparks, pinned checkpoint, temperature zero and thinking off. Coding asks for `merge_intervals` with a docstring and examples, capped at 200 completion tokens. Deterministic per-request prefixes and all category budgets match. The arithmetic mean covers eight categories; the counting ceiling is excluded.

Per-stream decode is `(completion_tokens - 1) / (total_seconds - first_token_seconds)`. Aggregate is total completion tokens divided by whole-batch wall time, including prefill and client scheduling. Completion tokens come from the server's usage block, not streamed chunk counts; speculative decoding can emit multiple tokens in a chunk. The first chunk can contain multiple tokens, so this is a shared benchmark convention rather than an exact measurement of steady-state token emission.

EP4 used a freshly restarted process and ran C1–C6 and C8 in sequence. The existing vLLM C1–C6 run used a fresh process; its C8 was measured separately with previously unused C8 prefixes. Each suite has three excluded warmup batches. This is a comparison of complete deployments with different engine/FlashInfer versions, parallelism and scheduling, not a controlled attribution to one source change. There are no repeated-run confidence intervals.

Across 261 measured category/counting requests, **261 prompt token counts match** and **112 output lengths differ**. The client-side prompt text is identical, but generated answers are not required to be identical. Timings and token counts do not establish task accuracy or quality parity. [Per-request counts](../results/ep4/matched-community/comparison.json).

## Coding at every concurrency

| C | vLLM decode/stream | EP4 decode/stream | vLLM aggregate | EP4 aggregate | vLLM TTFT (s) | EP4 TTFT (s) |
|---|---|---|---|---|---|---|
| 1 | 95.91 | 111.29 | 87.28 | 102.62 | 0.216 | 0.16 |
| 2 | 76.25 | 99.64 | 140.96 | 175.11 | 0.227 | 0.24 |
| 3 | 72.87 | 89.38 | 203.12 | 237.64 | 0.222 | 0.239 |
| 4 | 67.01 | 75.23 | 239.6 | 274.07 | 0.253 | 0.21 |
| 5 | 59.87 | 78.32 | 268.25 | 359.07 | 0.263 | 0.242 |
| 6 | 56.14 | 71.32 | 305.24 | 390.93 | 0.26 | 0.243 |
| 8 | 48.88 | 60.03 | 341.44 | 431.82 | 0.436 | 0.249 |

## Cold prefill

The unchanged benchmark requests one completion token and divides prompt tokens by total response time. Its raw field is named `ttft_s`, although the script uses total request time for these one-token cases.

| vLLM prompt tokens | EP4 prompt tokens | vLLM one-token time (s) | EP4 one-token time (s) | vLLM tok/s | EP4 tok/s |
|---|---|---|---|---|---|
| 2950 | 2950 | 1.059 | 1.187 | 2785.5 | 2485.0 |
| 11592 | 11592 | 4.008 | 3.715 | 2892.5 | 3120.1 |
| 46810 | 46810 | 15.844 | 12.606 | 2954.3 | 3713.4 |
| 93335 | 93335 | 33.505 | 21.905 | 2785.7 | 4261.0 |

All eight EP4 ranks completed without OOMs or restarts. One-second OS memory samples stayed above **26.63 GiB available** throughout restart and this benchmark. The post-run text/concurrency, one/four-image, JSON and tool checks passed. The earlier EP4 validation separately passed exact retrieval at 299,099 actual prompt tokens; this community prefill suite is shorter.

[Raw EP4 JSON](../results/ep4/matched-community/bench-ep4-matched-vllm.json) · [Raw EP4 tables](../results/ep4/matched-community/bench-ep4-matched-vllm.md) · [vLLM C1–C6](../../results/2026-09-11/bench-spark8.json) · [vLLM C8](../../results/2026-09-11/bench-spark8-c8.json) · [Post-run capability checks](../results/ep4/matched-community/acceptance/result.json)
