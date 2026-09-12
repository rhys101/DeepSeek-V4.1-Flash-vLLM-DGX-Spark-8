# Short-prompt concurrency: C8 through C128 with an 8M KV pool

These C8–C128 measurements belong to SG11. The later [SG17 RoCEnante result](sg17-rocenante-results.md) reached 134.91 coding decode tok/s at C1 and 474.52 coding aggregate tok/s at C8; SG17 has not had a C16–C128 throughput sweep.

**C128 passed on eight DGX Sparks: 1250.30 coding tokens/s aggregate and 774.68 prose tokens/s aggregate decode.**

Measured 12 September 2026 UTC. The principal table below uses one fresh 128-slot server configuration (SG11) for every concurrency level. TP8/EP4, native resident Engram, the checkpoint precision, the six production source overlays and the 8,000,000-token logical KV pool were retained. Earlier 32-slot and 64-slot runs are included separately below.

## Throughput

| Concurrency | Coding aggregate tok/s | Coding per-stream decode tok/s | Prose aggregate decode tok/s | Prose per-stream decode tok/s |
|---|---|---|---|---|
| C8 | 445.46 | 60.98 | 231.59 | 31.13 |
| C16 | 689.38 | 48.81 | 376.08 | 25.16 |
| C32 | 975.92 | 35.07 | 553.14 | 18.58 |
| C64 | 1216.86 | 21.89 | 698.20 | 11.84 |
| C128 | 1250.30 | 11.15 | 774.68 | 6.57 |

Coding is the repository’s unchanged community merge-intervals prompt with a 200-token output budget. Aggregate throughput divides server-reported completion tokens by full batch wall time, including prefill and request startup. Per-stream decode uses each request’s completion tokens minus one over its post-first-token response time.

Prose is the pinned sparkDash hash-map explanation prompt, with exactly 256 output tokens per stream. Its aggregate sums completion tokens minus one and divides by the earliest first-content to latest last-content window. Table values average two trials. These prose and coding totals have different workload and timing definitions; they are not directly interchangeable.

The pinned prose runner only allowed concurrency through C32. For SG10 and SG11, `--extended-concurrency` makes a temporary, hash-verified runtime copy and changes only the allowed-concurrency set to add C64 and C128. Prompts, token budgets, warmup, request execution and metric calculations remain unchanged. The original pinned files remain untouched; before/after hashes and the exact changed line are recorded in each result. [Extension implementation](../bench/sparkdash/concurrency-extension.mjs) · [Extension checks](../bench/sparkdash/test-concurrency-extension.mjs).

The initial SG10 prose attempt was rejected because the upstream runner omitted C64. That incomplete attempt is [preserved](../results/concurrency/sg10-ep4-8m-c64/runner-attempts.json) and excluded from the throughput tables. The successful community measurements were retained; only the prose suite was rerun with the explicit extension.

From C8 to C128 on this configuration, coding aggregate throughput changed by 2.81× and prose aggregate decode by 3.34×. Per-stream rates and first-token latency are reported alongside aggregate speed.

The final step from C64 to C128 changed coding aggregate throughput by **+2.7%** and prose aggregate decode by **+11.0%**. Coding per-stream decode changed from 21.89 to 11.15 tokens/s; prose changed from 11.84 to 6.57. Higher concurrency should be assessed against this per-request cost.

## Time to first token

| Concurrency | Coding mean TTFT (s) | Coding maximum TTFT (s) | Prose mean TTFT (s) | Prose maximum TTFT (s) |
|---|---|---|---|---|
| C8 | 0.215 | 0.216 | 0.228 | 0.237 |
| C16 | 0.406 | 0.422 | 0.347 | 0.351 |
| C32 | 0.636 | 0.779 | 0.554 | 0.682 |
| C64 | 1.057 | 1.533 | 0.871 | 1.306 |
| C128 | 1.906 | 3.304 | 1.543 | 3.268 |

These are client-observed first-output times. Prose means average the two trials; prose maxima are the largest individual first-token latency across both trials.

## Input lengths and pool size

Concurrent decode prompts contained **30–303 actual tokens per request**. Separate C1 prefill probes reached **93,335 input tokens**. Every performance prompt was below 128,000 tokens.

The shared logical KV pool remained configured at 8M tokens, but these short requests did not fill it. This experiment measures request concurrency and output throughput. It does not establish 128 simultaneous 1M-token contexts, or throughput at 128K input length. The distinct eight-context capacity test is [documented separately](eight-million-token-results.md).

## Eight-category workload average

| Concurrency | Mean aggregate tok/s | Mean per-stream decode tok/s | Mean TTFT (s) |
|---|---|---|---|
| C8 | 290.63 | 42.68 | 0.323 |
| C16 | 444.36 | 33.55 | 0.525 |
| C32 | 612.00 | 23.50 | 0.823 |
| C64 | 746.32 | 14.48 | 1.361 |
| C128 | 814.86 | 7.64 | 2.397 |

Each category runs in its own wave. The headline is the arithmetic mean of coding, JSON, narrative, prose, math, reasoning, summary and formatting rates; the counting ceiling is excluded. This is not a simultaneous mixture of request types. Most category budgets are 200 output tokens, summary is 150, and counting is 256; natural stopping is retained.

## Repeated prose trials

| Concurrency | Trial 1 aggregate decode tok/s | Trial 2 aggregate decode tok/s |
|---|---|---|
| C8 | 231.57 | 231.62 |
| C16 | 376.12 | 376.04 |
| C32 | 553.77 | 552.51 |
| C64 | 696.69 | 699.71 |
| C128 | 771.78 | 777.58 |

## Earlier slot configurations

| Run | Configured slots | Tested concurrency | Coding aggregate tok/s | Prose aggregate decode tok/s | Minimum available GiB per Spark |
|---|---|---|---|---|---|
| SG9 | 32 | C8 | 451.78 | 234.01 | 22.62 |
| SG9 | 32 | C16 | 688.15 | 379.74 | 22.62 |
| SG9 | 32 | C32 | 901.88 | 551.67 | 22.62 |
| SG10 | 64 | C32 | 969.19 | 555.57 | 18.09 |
| SG10 | 64 | C64 | 1152.54 | 706.62 | 18.09 |
| SG11 | 128 | C8 | 445.46 | 231.59 | 6.89 |
| SG11 | 128 | C16 | 689.38 | 376.08 | 6.89 |
| SG11 | 128 | C32 | 975.92 | 553.14 | 6.89 |
| SG11 | 128 | C64 | 1216.86 | 698.20 | 6.89 |
| SG11 | 128 | C128 | 1250.30 | 774.68 | 6.89 |

These rows preserve the earlier measurements rather than selecting only the fastest run. Changing the maximum request slots also changes graph captures and their memory overhead. Use the SG11 table for scaling across C8–C128 on one configuration; compare adjacent rows within SG9 or SG10 for those earlier configurations.

## Correctness and memory

All runs passed text arithmetic, two C8 arithmetic waves, single-image and four-image checks, structured JSON and a tool-call round trip. Additional exact arithmetic passed at C16 and C32 in SG9, C64 in SG10 and C128 in SG11. All eight containers were running at collection with no OOM kills or restarts. In SG11, the lowest sampled OS `MemAvailable` was **6.89 GiB on an individual Spark**; the guard threshold was 2 GiB per Spark.

Fresh API snapshots recorded each concurrency with an 8M token capacity, and server decode logs independently recorded actual running batches at each level. Memory sampling covers startup and testing at approximately one-second intervals, supplemented by the all-rank guard. Sampling cannot rule out lower values between observations. These correctness smokes do not establish broad quality parity.

| Run | Lowest available memory on any Spark (GiB) | Maximum additional OS swap use on any Spark (GiB) |
|---|---|---|
| SG9 | 22.619 | 0.000 |
| SG10 | 18.090 | 0.069 |
| SG11 | 6.886 | 0.000 |

Swap figures are the largest increase from the first benchmark-guard sample on any one Spark. They are whole-system counters and do not identify which process used swap.

The SG11 minimum of 6.886 GiB occurred at 01:01:21 UTC during startup graph capture. The benchmark guard, started after readiness, recorded a minimum of 8.714 GiB. [SG9 minimum samples](../results/concurrency/sg9-ep4-8m-c32/memory-minima.json) · [SG10 minimum samples](../results/concurrency/sg10-ep4-8m-c64/memory-minima.json) · [SG11 minimum samples](../results/concurrency/sg11-ep4-8m-c128/memory-minima.json).

## Configuration and reproduction

| Setting | SG11 concurrency experiment |
|---|---|
| Hardware / parallelism | Eight GB10/SM121 DGX Sparks; TP8, EP4, MoE-TP2 |
| Model | deepseek-ai/DeepSeek-V4.1-Flash |
| Model revision | df42c109f1defefcbfcedbe7d905718a12266e40 |
| Engram | Native resident owned rows |
| Context limit / shared logical pool | 1,000,000 per request / 8,000,000 total |
| Maximum running requests | 128 |
| Prefill chunk / max-prefill tokens | 2,048 / 2,048 |
| Static memory fraction | 0.80 |
| Speculation | DSpark five-token drafting |
| Graph batch sizes | 1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 24, 32, 48, 64, 96, 128 |
| OS reserve guard | 2 GiB per Spark |

The model precision remains checkpoint MXFP4 experts, FP8 dense weights, BF16 activation dtype and the existing FlashInfer expert compute path; automatic KV selection resolves to FP8 E4M3. The production overlay hashes and the image ID are recorded with each run. [Source composition and precision details](build-and-pins.md).

The packaged public launcher still validates the standard SG5 profile (300K context, 3.2M pool, eight request slots). These results come from separately configured experimental launches; editing only the example JSON is insufficient because the launcher validates its configuration. The exact experimental settings are included in the evidence, while the README’s standard build and launch guide remains the packaged deployment.

After configuring a compatible experimental server, the benchmark CLIs used for the SG11 table are:

```bash
python3 bench/v41bench.py --base http://HEAD:8000/v1 --model deepseek-v41-flash \
  --label sg11-8m-short --out NEW_COMMUNITY_DIR --levels 8,16,32,64,128 \
  --prefill 2000,8000,32000,64000
node sglang/bench/sparkdash/run.mjs --base http://HEAD:8000/v1 \
  --model deepseek-v41-flash --label sg11-8m-short --out NEW_PROSE_DIR \
  --extended-concurrency --trials 2 --sizes 4096,16384,65536 --levels 8,16,32,64,128
```

Create the community output directory first. Run the community suite once after a fresh server start: its deterministic prefixes repeat across runs. The sparkDash runner uses its pinned upstream warmup and prompt reuse unchanged. Requests use temperature zero and thinking off. All token counts come from API usage, not SSE chunk counts. Client runtime was Node v22.22.2 for sparkDash.

## Evidence

- [Repository and runner validation](../results/concurrency/runner-validation.json)
- [SG9 summary](../results/concurrency/sg9-ep4-8m-c32/summary.json), [community requests](../results/concurrency/sg9-ep4-8m-c32/community.json), [prose metrics](../results/concurrency/sg9-ep4-8m-c32/sparkdash-metrics.json), [configuration](../results/concurrency/sg9-ep4-8m-c32/configuration.json), [concurrency proof](../results/concurrency/sg9-ep4-8m-c32/concurrency-evidence.json), [correctness](../results/concurrency/sg9-ep4-8m-c32/correctness.json), [server logs](../results/concurrency/sg9-ep4-8m-c32/server-decode.log), [provenance](../results/concurrency/sg9-ep4-8m-c32/provenance.json)
- [SG10 summary](../results/concurrency/sg10-ep4-8m-c64/summary.json), [community requests](../results/concurrency/sg10-ep4-8m-c64/community.json), [prose metrics](../results/concurrency/sg10-ep4-8m-c64/sparkdash-metrics.json), [configuration](../results/concurrency/sg10-ep4-8m-c64/configuration.json), [concurrency proof](../results/concurrency/sg10-ep4-8m-c64/concurrency-evidence.json), [correctness](../results/concurrency/sg10-ep4-8m-c64/correctness.json), [server logs](../results/concurrency/sg10-ep4-8m-c64/server-decode.log), [provenance](../results/concurrency/sg10-ep4-8m-c64/provenance.json)
- [SG11 summary](../results/concurrency/sg11-ep4-8m-c128/summary.json), [community requests](../results/concurrency/sg11-ep4-8m-c128/community.json), [prose metrics](../results/concurrency/sg11-ep4-8m-c128/sparkdash-metrics.json), [configuration](../results/concurrency/sg11-ep4-8m-c128/configuration.json), [concurrency proof](../results/concurrency/sg11-ep4-8m-c128/concurrency-evidence.json), [correctness](../results/concurrency/sg11-ep4-8m-c128/correctness.json), [server logs](../results/concurrency/sg11-ep4-8m-c128/server-decode.log), [provenance](../results/concurrency/sg11-ep4-8m-c128/provenance.json)

These are short benchmark waves: one community wave per category/concurrency and two prose waves. They are not a long-duration saturation test or a broad workload-quality evaluation.
