# EP4 validation and limits

The initial EP4/SG5 serving run completed on 11 September 2026 with global TP8 and EP4/MoE-TP2, DSpark block size five, eight requests, four images and 300K configured context. [Source identity](build-and-pins.md).

## Quality

| Check | Result and scope |
|---|---|
| Text and concurrency | Arithmetic passed at C1 and in two C8 waves |
| Vision | Printed code, colored shapes and four-image ordering passed |
| Structured output | Strict JSON schema fixture passed |
| Tools | One requested function call, arguments and returned-value round trip passed |
| Long context | Three synthetic records retrieved exactly at 32,867 / 131,171 / 299,099 actual prompt tokens |
| Memory and process health | At least 17.5744 GiB OS-available on every node in one-second samples; no OOMs or restarts |

The retrieval runs took 7.090, 32.884 and 103.783 seconds respectively. [Capability result](../results/ep4/acceptance/result.json), [long-context result](../results/ep4/long-context/result.json), [per-rank memory minima](../results/ep4/sparkdash-summary.json).

These are narrow capability checks. They do not establish broad answer-quality or reasoning parity, task completion accuracy, eight simultaneous 300K requests, or equivalent behavior on arbitrary images/tools. Benchmarks use thinking off and short output budgets. Format preservation alone is not a proof of numerical equivalence.

## Expert partition and draft context

The EP4 probe passed **90 cases** using real layer-0 target weights and all three draft-stage expert weights, across all four expert groups. It checks mixed/local/nonlocal routes, clamp-sensitive inputs, target sizes through 9,215 rows, draft batches, changed-input graph replay and exact repeated outputs. Concatenating the split packed weights/scales reconstructs the original bytes exactly.

On one idle SM121 GPU, it simulates each group's two TP ranks and compares their summed outputs with two EP8 ranks and the unsplit group. The maximum relative L2 among these comparisons was **0.003395**, below the prespecified 0.02 limit; cosine had to be at least 0.9995. Measured replay allocation growth was zero. This is local partition arithmetic, not a distributed communication test, an FP32 full-model oracle or a broad quality score. Splitting experts changes floating-point partial sums. [Results](../results/ep4/partition-check/result.json), [probe](../results/ep4/partition-check/probe-source.py).

All eight startup receipts confirm forty target and three draft FlashInfer CUTLASS modules with the expected EP4/MoE-TP2 counts and shapes. [Loaded layouts](../results/ep4/cluster/selected-moe-layout.json). The draft-context fixture checks explicit/inherited selection, optional draft parallel context, nested scopes and restoration on exceptions. [Context result](../results/ep4/context-test-result.json).

## Unchanged attention and verification operators

The exact five verification/index files, unchanged from SG3, passed **36 tests and four subtests** on SM121, with four all-padded cases skipped by the upstream suite. Coverage includes decode/verify replay, rejected prefixes, ring wrap, changed-input CUDA graph replay, real compressed KV/index writes, candidate masks, ties and non-finite candidates. Tolerances were unchanged. [Recorded output](../results/sg3/kernel-preflight/check.log), [JUnit](../results/sg3/kernel-preflight/junit.xml), [test source](../validation/upstream-tests/).

The native-head suite passed twelve upstream H8/H16 numerical subcases, model head/sink selection checks and eight H8 graph cases at token counts 1/6/24/48 with single/dual caches. Each graph case replayed three changed inputs. The comparison uses the upstream absolute/relative tolerance of 0.05; some maximum absolute errors are large because the fixture generates large-magnitude synthetic outputs. These are not bitwise-equivalence tests or model-quality scores. [Recorded native-head result](../results/sg3/native-head-check/result.json), [log](../results/sg3/native-head-check/check.log).

An earlier, broader ten-file PR test is not counted as evidence for this narrower deployed composition. The original tests' hardware skips and deprecation warnings are retained in their logs.

The packaged EP4 image passed build and all eight final-source checks. Its source-extracted draft-context fixture passed eight cases. The seven unchanged operator files reuse the immediately preceding packaged-image GPU results: native-head checks and 36 verification tests/four subtests passed, with four upstream skips. [Build and validation evidence](../results/ep4/package-validation/). These reused results qualify the unchanged operators; the separate 90-case probe and full serving run qualify EP4 partitioning.

The public CPU tests verify all eight launch configurations, reject changes outside the EP4 profile, reject placeholder image IDs, check actual all-rank layout receipts, exercise draft-context restoration, and ensure overlay failures occur before files are replaced. Ordinary GitHub CI has no Spark GPU; hardware results are recorded separately.

## Speed

The [completed engine comparison](community-comparison.md) uses the unchanged community coding/category workload: 63 batches, 261 requests and four cold-prefill cases. All prompt token counts match the existing vLLM results. Post-run capability checks passed, and all eight ranks retained the expected [actual EP4 layout](../results/ep4/matched-community/selected-moe-layout.json), without OOMs or restarts. The separate sparkDash prose run has two trials, with first-pass prefill results retained. Against SG3, EP4 improves C1/C4/C6 and loses 2.26% at C8; no confidence intervals are established. [Prose results](prose-results.md).

Published raw measurements retain timings, token counts, prompt hashes and source identities. Operational server addresses and output directories in the sparkDash envelope are replaced with neutral labels; hardware-specific cluster configuration, SSH details and full container dumps are excluded.
