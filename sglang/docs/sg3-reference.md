# SG3 sparkDash prose and prefill results

Validated 11 September 2026, 18:04 UTC. This candidate adds five exact production files from [SGLang #39068](https://github.com/sgl-project/sglang/pull/39068) over SG2b: fixed-width ratio-2 verification compression, index-page postprocessing and candidate-mask publication. It retains native query heads, `--min-free-slots-delay 1`, TP8/EP8, DSpark 5, all precision settings, eight requests, four images and 300K context.

## Speed

| Concurrent requests | Trial 1 / 2, aggregate tok/s | Mean | Change from SG2b | Maximum TTFT, trial 1 / 2, ms |
|---|---:|---:|---:|---:|
| 1 | 63.91 / 64.13 | 64.020 | +0.68% | 160.20 / 153.48 |
| 2 | 100.46 / 101.11 | 100.785 | +2.79% | 212.26 / 213.69 |
| 3 | 130.43 / 131.30 | 130.865 | +0.28% | 207.71 / 205.94 |
| 4 | 148.50 / 149.07 | 148.785 | +0.27% | 199.85 / 198.80 |
| 6 | 176.41 / 178.35 | 177.380 | +0.68% | 212.29 / 223.70 |
| 8 | 233.12 / 234.73 | 233.925 | +0.27% | 464.44 / 406.50 |

C1 and C4 are **1.3768× / 1.4279×** Mia's reported four-Spark figures of 46.5 / 104.2 tok/s. The requested lower target, 1.5×, would require 69.75 / 156.30 tok/s. Reaching those thresholds would require a further 8.95% / 5.05% increase from SG3. Improvements over SG2b at C1/C4/C8 are below 1% across two trials; this is insufficient evidence of a robust gain beyond normal variation. Mia's exact benchmark software/image is unknown.

| Prefill target | First pass tok/s | Second pass tok/s |
|---|---:|---:|
| 4,096 | 2910.43 | 4500.31 |
| 16,384 | 2630.42 | 4765.36 |
| 32,768 | 4147.56 | 4763.18 |
| 65,536 | 4351.07 | 4632.75 |
| 131,072 | 3958.67 | 4151.78 |

The slower first pass is retained. Unique prefix salts prevent prefix-cache reuse; warmed kernels/allocators remain a separate effect. All upstream sparkDash benchmark files are pinned and unmodified; see [raw results](../results/sg3/sparkdash/result.json).

Admission diagnostics: idle C8 TTFT 496–506 ms; immediate-after-warmup seven requests at 259–260 ms and the eighth at 430 ms; one-second-after-warmup all at 1.25–1.27 seconds. The single delayed eighth request lasting several seconds is resolved in the matched benchmark, but these results do not establish uniformly low TTFT for every arrival pattern.

## Quality and correctness

Text, C8 arithmetic, one/four-image checks, structured JSON and tool round trip passed. Exact three-record retrieval passed at 32,867, 131,171 and 299,099 actual prompt tokens, taking 7.116 / 31.984 / 98.432 seconds. These are capability smokes; broad quality parity remains unmeasured. Precision formats, acceptance thresholds, reasoning mode and draft length were unchanged. C1's saved output previews and character counts match SG2b; previews are not complete token-identity evidence.

The exact five-file composition passed 36 upstream tests and four subtests on GB10/SM121; four all-padded cases were skipped by the upstream suite. Tests cover verify/decode replay, ring wrap, rejected prefixes, changed-input graph replay, real compressed KV/index writes, page mapping and candidate masks including ties/non-finite values. Tolerances were unchanged. See [the raw kernel preflight](../results/sg3/kernel-preflight/check.log).

Every node retained at least **19.86 GiB** OS-available memory in one-second samples through validation. All eight ranks had no OOMs or restarts. This does not certify eight simultaneous 300K requests.

This table uses sparkDash’s prose prompt and aggregate decode timing. Use the matched community coding/category table in the main SG3 report to compare against the published vLLM numbers. [Build/source identity](build-and-pins.md).
