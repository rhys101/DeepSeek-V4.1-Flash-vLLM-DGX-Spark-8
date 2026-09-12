# SG17: 134.91 coding tokens/s on eight DGX Sparks

**134.91 tok/s single-request coding decode · 474.52 tok/s coding aggregate at C8 · +22.21% C1 decode versus the latest SG11 control.**

Measured 12 September 2026 UTC. SG17 averaged **134.680 tok/s** in its initial five measured C1 trials and **134.906 tok/s** in a complete repeat after long-context and capability validation on the same process. All ten measured trials exceeded 130 tok/s. The repeat range was **134.87–134.98 tok/s**. These are means of the benchmark's recorded two-decimal trial rates. [Selection receipt](../results/sg17-rocenante/selection.json).

## Serving measurements

| Concurrency | Coding decode per stream (tok/s) | Coding full-batch aggregate (tok/s) | Prose aggregate decode (tok/s) |
|---|---|---|---|
| C1 | 134.91 | 121.86 | 80.99 |
| C4 | 83.67 | 305.00 | 180.46 |
| C8 | 64.89 | 474.52 | 244.15 |

The table uses the second complete SG17 benchmark. The initial run averaged **134.68 / 84.00 / 64.60 tok/s per coding stream** at C1/C4/C8, with **122.66 / 305.87 / 470.58 tok/s coding full-batch aggregate**, respectively. All runs, including the earlier SG11 controls, are retained in the [comparison](../results/sg17-rocenante/comparison.json) and [per-request evidence](../results/sg17-rocenante/runs/).

| Run (chronological order) | C1 coding decode (tok/s) | C1 full-request (tok/s) | C1 prose decode (tok/s) |
|---|---|---|---|
| sg11_fresh_before | 110.858 | 102.286 | 66.947 |
| sg11_later_before | 108.768 | 99.082 | 66.663 |
| sg11_recent_fresh | 111.002 | 102.562 | 67.060 |
| sg11_recent_postlong | 110.390 | 100.104 | 66.163 |
| sg17 | 134.680 | 122.658 | 81.047 |
| sg17_repeat | 134.906 | 121.860 | 80.993 |

The latest SG11 fresh control averaged 111.002 tok/s; its same-process post-long-context control averaged 110.390 tok/s. **Both were measured before SG17.** The SG17 repeat is 22.21% above that last control. Earlier SG11 measurements show modest drift. There is no SG11 control after SG17 and no extended soak behind this result.

Measured C1 coding trials, in order:

| SG17 run | Trial 1 | Trial 2 | Trial 3 | Trial 4 | Trial 5 | Mean |
|---|---|---|---|---|---|---|
| Initial | 135.26 | 135.26 | 135.17 | 135.25 | 132.46 | 134.680 |
| After long context | 134.87 | 134.98 | 134.92 | 134.87 | 134.89 | 134.906 |

## Measurement procedure

Coding uses the repository's unchanged [community benchmark](../../bench/v41bench.py), with the merge-intervals prompt, 200 completion tokens, temperature zero and thinking disabled. These requests had 47 input tokens. Each suite runs the standard warmups and one excluded full coding batch at each concurrency, then retains five C1 and three C4/C8 batches; the first three trials alternate C1/C4/C8 ordering. The excluded warmups remain in each `coding.json`; the initial SG17 C1 coding warmup was 129.96 tok/s and is not one of the ten measured trials. Prefixes are deliberately reused and no cache flush is performed. This is warm decode testing, not a cold-request benchmark.

Coding per-stream decode divides completion tokens minus one by post-first-token response time. Coding aggregate divides completion tokens by full batch wall time, including prefill and startup. Prose uses the pinned sparkDash hash-map prompt, 256 completion tokens per stream and three trials per concurrency. Its aggregate divides the sum of completion tokens minus one by the earliest-first-content to latest-last-content interval. These distinct workload and timing definitions are reported separately. The prose C64/C128 allowlist extension is not enabled here.

The first SG17 suite follows capability checks and the C128 arithmetic burst. Exact retrieval at three context lengths and another capability pass precede the repeat. The [portable benchmark wrapper](../experiments/sg17-rocenante/benchmark-client.py) preserves the measured procedure and changes only endpoint/file locations to repository paths. The community source SHA-256 is `e0d6b2d25bd585d11fbdf39c2ddcdf7a4de8ab685af6bd42465e69f3ee6e80a8`; prose source is sparkDash `d0c7f71296a1071d0d75f95b14c21413d4d06321`.

## What changed

SG17 retains **TP8/EP4, native resident Engram, an 8,000,000-token shared logical KV pool, 128 request slots, a 1,000,000-token context cap, 2,048-token prefill chunks, static memory fraction 0.80 and DSpark block size five**. Both target and draft MoE use FlashInfer MXFP4. Checkpoint MXFP4 experts, FP8 dense weights, BF16 activation dtype and FP8 E4M3 KV selection remain as before. [Recorded configuration](../results/sg17-rocenante/configuration.json).

Every serving engine argument matches SG11 on all eight ranks, and its six production overlays remain byte-identical. Three additional SGLang overlays route eligible global-TP8 BF16/FP32 SUM reductions of at most 512 KiB through RoCEnante, connect its CUDA graph capture context and check transport health before processing completed output. Other process groups, large reductions, FP16, non-SUM operations and all-gathers retain their existing routes. [Argument parity](../results/sg17-rocenante/argument-parity.json) · [Exact source changes](../experiments/sg17-rocenante/sglang-rocenante.patch).

The frozen source bundle combines the previously qualified B12x kernels/compiler with eight RoCEnante files from `081b235931dbbcedcf0eb5899bae990c5dec5238` and two registry changes. Its communication buffers use pinned host memory accessible to the GPU and NIC. Engram remains resident on each GPU's unified memory; this adds no host Engram-table offload. [Source bundle, manifests and configuration](../experiments/sg17-rocenante/).

RoCEnante accumulates in FP32 in fixed rank order; NCCL's BF16 rounding differs. **SG17 prose output previews differ from SG11.** Coding text was not retained, so matching token or character counts do not establish identical output. The evidence supports faster performance on these benchmarks; it does not establish broad quality equivalence or attribute every part of the throughput gain solely to transport latency.

## Validation and memory

Text, repeated C8 arithmetic, one-image and four-image requests, strict JSON and a tool-call round trip passed before and after the initial benchmark. **128/128 concurrent arithmetic requests** returned the expected integers. Exact three-record retrieval passed at **32,866, 131,170 and 299,098 input tokens**, taking 7.70, 35.71 and 108.10 seconds. [Correctness](../results/sg17-rocenante/correctness.json) · [Long-context cases](../results/sg17-rocenante/long-context.json).

The lowest recorded OS-available reserve on any Spark was **5.121 GiB**, with a **2 GiB guard floor**. No OOM kills or container restarts occurred. The largest sampled OS swap increase was **6.582 GiB** within a recorded phase; [the per-rank memory receipt](../results/sg17-rocenante/memory-summary.json) separates startup/validation from the clean repeat. Whole-system swap counters do not identify the process responsible. Sampling cannot rule out lower reserve between observations.

At **13:08:52 UTC**, the final receipt showed all eight SG17 containers running, an idle healthy API, correct source/image identity, native Engram, target/draft EP4 geometry and active RoCE transport markers. It checked nine overlays plus runtime configuration and all 241 B12x bundle files per rank. [Dated health receipt](../results/sg17-rocenante/health.json). This is a recorded observation, not a live status endpoint.

These checks do not establish an occupied 8M pool, a full 1M-token request, broad model quality or SG17 throughput at C16–C128. The earlier [SG11 concurrency sweep](concurrency-results.md) and [eight-context capacity test](eight-million-token-results.md) describe separate configurations and runs.

## Eight-GPU transport qualification

Before model serving, the unchanged upstream GPU suite passed **74 tests with one skip per rank**. The skipped 6 MiB int64 all-gather exceeds its 4 MiB fixture capacity; all-gather is outside the SG17 route. [Per-rank upstream results](../results/sg17-rocenante/component/upstream-tests.json).

The custom component probe passed 27 dtype/shape/seed cases with in-place and out-of-place checks against a fixed-order FP32 oracle, nine dyadic cases and 24 changing-input graph replays on each rank. The matching copy-plus-reduction graph benchmark measured NCCL/RoCEnante latency ratios of **3.56× / 3.54× / 3.41× / 2.39× / 1.83×** for token dimensions 1/5/6/24/48. These are collective latency ratios without model weights resident, not end-to-end inference speedups. [Component comparison](../results/sg17-rocenante/component/comparison.json) · [Raw per-rank intervals](../results/sg17-rocenante/component/microbenchmark.json).

The exact SGLang integration passed 24 mixed-operation graph replays per rank, with five RoCE operations per replay and NCCL fallback coverage. Stopping a proxy during a real replay caused the actual scheduler decode and idle result handlers to raise before output processing; the eventless health path also raised. All eight final integration containers exited cleanly. [Integration evidence](../results/sg17-rocenante/component/integration.json).

Harness corrections are part of the record: the first upstream collector encountered a root-owned XML permission error after the tests passed; the first component benchmark aliased the FP32 reference input and was fixed by cloning it; integration v1 incorrectly treated an allocation decrease as failure, while v2 passed its checks but hung in teardown with live CUDA graph references. The published qualification uses the corrected component v2 and clean-exit integration v3. These earlier attempts are excluded from passing performance evidence.

## Evidence provenance

The public extracts retain all comparison trials, excluded coding warmups, per-request timings/token counts, prose previews, source identity, capability results and dated health/memory receipts. Connection addresses, private filesystem paths and hardware UUID snapshots are omitted. [Source and published-file hashes](../results/sg17-rocenante/provenance.json).
