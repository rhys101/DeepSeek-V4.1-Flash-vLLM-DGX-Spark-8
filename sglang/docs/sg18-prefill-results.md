# SG18 native prefill: promoted results

Measured **14 September 2026**, promoted **15 September 2026**. The native TP8
prefill split passed the unchanged cold and README confirmation against both
scratch-fixed controls, then passed the requested capacity repeat after a
Spark1 reboot. The exact source is active on all eight Sparks.
[Promotion record](../results/sg18-prefill-tp-split/promotion.json) ·
[Source and integration](../experiments/sg18-prefill-tp-split/).

## Cold prefill

| Concurrency | Input tokens each | Mean seconds | Aggregate input tok/s | All three measured times, seconds |
|---|---:|---:|---:|---|
| C1 | 4,096 | 0.92493 | 4,428.42 | 0.94206, 0.91301, 0.91973 |
| C1 | 32,768 | 7.80877 | 4,196.31 | 7.77461, 7.74932, 7.90238 |
| C1 | 131,072 | 33.54887 | 3,906.90 | 34.18924, 33.17145, 33.28591 |
| C1 | 299,008 | 86.46150 | 3,458.28 | 88.07243, 85.83269, 85.47937 |
| C8 | 32,768 | 61.87907 | 4,236.39 | 62.73461, 61.38802, 61.51458 |
| C8 | 131,072 | 265.86832 | 3,943.97 | 269.31916, 264.04349, 264.24231 |

These are all six cells of `prefill-native-confirmation-v01`. Each has one
excluded warmup and three measured trials; no slow trial is discarded. The
client flushes the cache, submits distinct frozen prompts and requests one
generated token per request. Time ends at the last request's first token.
Rates divide total input tokens by the mean time, rather than averaging rates.
All reported cached-token counts were zero. Each control and the candidate
started on a fresh serving process with zero previous prefill.
**65,536-token prompts were not measured; no 64K interpolation is claimed.**

| Primary long workload | Fixed before, seconds | Candidate, seconds | Fixed after, seconds |
|---|---:|---:|---:|
| C1 × 131,072 | 42.67493 | 33.54887 | 39.27536 |
| C1 × 299,008 | 149.11441 | 86.46150 | 128.45972 |
| C8 × 131,072 each | 325.34942 | 265.86832 | 314.72990 |

The geometric mean time across these cells fell **28.05% against the preceding
control** and **21.40% against the following control**. No short-cell mean
regressed in this confirmation. The controls are reported separately, and the
candidate source and gates remained unchanged. These sequential blocks do not
establish a universal speedup or identical thermal/allocator state.

The earlier complete comparison had inconsistent control timing and a C8/32K
regression of 5.08% against its preceding control. Its gate remained unmet.
It is retained separately, together with every warmup, slower trial and the
interrupted attempt; it is not averaged into the accepted confirmation.
[Historical evaluation](../results/sg18-prefill-tp-split/evaluation.md) ·
[All recorded measurements](../results/sg18-prefill-tp-split/measurements.json).

## README decode suite

| Run | Concurrency | Coding decode per stream, tok/s | Coding full-batch aggregate, tok/s | Prose aggregate decode, tok/s |
|---|---|---:|---:|---:|
| initial | C1 | 131.33 | 120.14 | 86.88 |
| initial | C4 | 84.45 | 305.88 | 168.06 |
| initial | C8 | 68.15 | 498.06 | 255.20 |
| repeat | C1 | 131.72 | 119.86 | 87.30 |
| repeat | C4 | 84.24 | 306.08 | 168.47 |
| repeat | C8 | 69.78 | 508.79 | 254.65 |

Headlines use the repeat suite, as in the previous README. Coding uses the
unchanged Tony benchmark with 200 output tokens; prose uses the pinned
sparkDash runner with 256. Coding C1 excludes first-token time; aggregate
coding includes the complete batch. Prose uses its original output window.
All coding/prose trial rates and statistics remain in the promotion JSON and
raw evidence. The pinned prose runner does not retain absolute aggregate-window
endpoints, so that window cannot be independently reconstructed from saved
per-stream durations. [Original benchmark method](sg18-indexer-mhc-wo-results.md#exact-readme-method).

Both README repetitions passed the C1/C8 comparison against both confirmation
controls, with no repeatable decline over 3%. All 47 answer comparisons matched
observed control variants. Both capability passes, image/JSON/tool checks,
128/128 simultaneous arithmetic requests and the three pinned long-context
retrievals passed. These remain bounded checks; the earlier SG18 quality
differences and its lower single-stream result than the SG17 record remain
in the [prior study](sg18-prior-study.md).

## Eight near-million-token contexts

The capacity repeat used the same eight distinct **997,100-token** documents as
the failed first attempt: **7,976,800 input tokens total**. It passed 8/8 cold
retrievals and 8/8 cached retrievals, with exactly 1,024 generated tokens per
cached request. Thirty fresh server samples showed eight running requests,
zero waiting and more than 7.9M used and active tokens simultaneously; the
maximum used-token count during those observations was **7,983,872**.

The cold batch took **4,500.87 seconds (75.01 minutes)**, or **1,772.28 input
tok/s** including its completed 20-token retrieval replies. The cached phase
took **60.32 seconds** for 8,192 output tokens; cached input tokens are not a
cold-prefill rate. Each cached request reported 996,864 cached input tokens.

The requested repeat guard was **1 GiB OS-available per node**. All 1,299 samples
passed; the minimum was **3.555 GiB on Spark7**, and Spark1's minimum was
**4.446 GiB**. Every sample retained all eight unchanged worker processes with
the same image, no recorded OOM and no restart. The maximum sampling interval
was 4.07 seconds; this is sampled evidence, not continuous observation.

The earlier attempt stopped at **1.953 GiB on Spark1**, crossing its then-fixed
2 GiB guard before any retrieval completed. That failure remains recorded.
This repeat stayed above 2 GiB in every sample, so it did not require the extra
headroom allowed by the relaxed threshold. A successful repeat after reboot
does not isolate the cause of the earlier drop. This establishes exact
synthetic retrieval and simultaneous residency, not broad million-token
reasoning quality. [Retest data](../results/sg18-prefill-tp-split/capacity-retest.json).

## Runtime, source and reproduction

The deployment retains TP8/EP4, resident Engram, static-five DSpark, 2,048-token
prefill chunks, 128 slots, a 1M context limit and an 8M logical KV pool. All 500
serving-setting checks passed. Final live audits verified all 261 staged files
and two expert caches per rank, healthy transport and logs, an idle service,
preserved stopped baselines and no remaining experiment jobs.
[Publication health audit](../results/sg18-prefill-tp-split/promotion-health.json).

Source qualification passed 45 native GPU cases per rank and 192 serving
decode-dispatch checks. Ordinary cases require exact indices and masks;
deliberately oversubscribed all-zero ties use the documented optimal-score
oracle because the original top-k also changes tied indices. GPU execution
is retained evidence; the public replay uses no GPU.

From the repository root:

```bash
python3 sglang/experiments/sg18-prefill-tp-split/verify-source.py
python3 sglang/experiments/sg18-prefill-tp-split/verify-results.py
python3 sglang/experiments/sg18-prefill-tp-split/verify-promotion.py
```

The verifiers check exact source and patch reconstruction, all historical
measurements including failures, and the successful retest with its promotion
headlines. [Archive contents and limitations](../results/sg18-prefill-tp-split/).
The standard Dockerfile and launcher still package SG5. The promoted measured
integration has its own source snapshot; a generalized public installer has
not been validated.
