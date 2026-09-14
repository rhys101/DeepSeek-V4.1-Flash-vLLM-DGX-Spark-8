# SG18: indexer, mHC and WO qualification

Measured **14 September 2026** on eight DGX Sparks. SG18's repeated README suite
averaged **131.05 coding decode tok/s at C1** and **495.32 coding full-batch
aggregate tok/s at C8**. Against the previously published SG17 means, those
changes are **-2.86%** and **+4.38%**. The initial SG18 means were
**131.28** and **505.25**, respectively.

The predeclared headline gate requires both candidate runs to exceed both
fresh SG17 controls and the published SG17 C1/C8 headlines. **Gate: NOT MET.**
[Exact gate inputs and all statistics](../results/sg18-indexer-mhc-wo/comparison.json).

SG18 is presented as the C8-throughput choice. Its C1 mean is 2.86% lower than the published SG17 record; SG17 remains the faster C1 reference. The stronger two-metric gate remains unmet, and no trial was changed to clear it.

## Every complete run

All rates are tok/s. Coding decode excludes first-token latency; full-batch
aggregate includes prefill. Prose uses its own first-to-last-output interval.

| Run | Concurrency | Coding decode per stream | Coding full-batch aggregate | Prose aggregate decode |
|---|---|---|---|---|
| sg17_initial | C1 | 132.44 | 120.41 | 79.96 |
| sg17_initial | C4 | 82.35 | 302.51 | 177.75 |
| sg17_initial | C8 | 63.66 | 462.24 | 241.01 |
| sg17_repeat | C1 | 133.48 | 117.39 | 76.59 |
| sg17_repeat | C4 | 81.76 | 292.20 | 178.81 |
| sg17_repeat | C8 | 64.62 | 465.76 | 237.89 |
| sg18_initial | C1 | 131.28 | 119.38 | 87.38 |
| sg18_initial | C4 | 84.45 | 302.78 | 169.23 |
| sg18_initial | C8 | 70.46 | 505.25 | 255.98 |
| sg18_repeat | C1 | 131.05 | 117.49 | 86.89 |
| sg18_repeat | C4 | 83.22 | 299.71 | 166.76 |
| sg18_repeat | C8 | 68.33 | 495.32 | 253.84 |

The tradeoff also includes C4 prose: **166.76 tok/s** in the SG18 repeat
versus **180.46 tok/s** in the published SG17 repeat
(**-7.59%**). Prose improved at C1/C8. C4 coding
full-batch aggregate was also below the published SG17 mean. These workloads
are reported individually rather than collapsed into a single speed claim.

Each coding run contains five C1 trials and three C4/C8 trials; prose contains
three trials at each concurrency. All trial values, minima, maxima and sample
standard deviations are in the comparison JSON. Every measured coding request,
all excluded full coding warmups, and per-stream prose metrics are retained in
the [four run directories](../results/sg18-indexer-mhc-wo/runs/).
No slower trial or outlier was removed. These are sequential launch blocks,
not randomized alternation or evidence of a universal kernel speedup.

## Exact README method

The client uses the unchanged [SG17 wrapper](../experiments/sg17-rocenante/benchmark-client.py)
at repository commit `66aa6bd5085e63bbb011557075765e3191e807b2` and unchanged
[Tony benchmark](../../bench/v41bench.py), source SHA-256
`e0d6b2d25bd585d11fbdf39c2ddcdf7a4de8ab685af6bd42465e69f3ee6e80a8`.
Coding uses the same merge-intervals prompt, deterministic prefixes, 47 input
tokens, 200 output tokens, temperature zero and thinking off. Decode is
`(completion_tokens - 1) / (total_seconds - first_token_seconds)`;
aggregate is total completion tokens divided by complete batch wall time.
Means use the runner's published two-decimal trial rates, matching SG17.

Each run starts with the original three 64-token warmup categories and one
excluded full coding batch at C1/C4/C8. Trial two reverses the coding concurrency
order to C8/C4/C1; prefixes are reused and no cache flush is inserted.
Prose uses the unchanged pinned sparkDash runner, its hash-map prompt and
256-token budget: 32 input tokens at C1, 39 at C4/C8 because of the original
stream suffix. Three 4K prefill probes are retained with each prose run.
The [35-file reference manifest](../results/sg18-indexer-mhc-wo/reference-manifest.json)
was verified before each phase and after completion.

On both original and candidate, the sequence was capability checks, 128-way
arithmetic, initial benchmark, the original SG17 three-record retrieval
requests, capability checks again, then the repeat benchmark. The same original
long-context tag and prompt hashes were used. Fresh candidate startup additionally
passed source/runtime and captured-dispatch checks before the suite.

Independent validation checked exact prompts and token counts, coding decode
arithmetic, coding aggregate against the precision of the retained wall time,
prose per-stream arithmetic, all expected trials, and all capability answers.
The pinned prose runner does not retain absolute aggregate-window endpoints;
its aggregate cannot be independently reconstructed from the saved per-stream
durations alone. It is reported under the unchanged runner's original method.
An initial validator assumption incorrectly expected 32 prose input tokens at
every concurrency. This was corrected to match the existing published 32/39-token
fixtures; the benchmark source, data and trial selection were unchanged.

## First-token latency

| Run | Concurrency | Coding mean TTFT (s) | Prose mean TTFT (ms) |
|---|---|---|---|
| sg17_initial | C1 | 0.158 | 147.33 |
| sg17_initial | C4 | 0.180 | 167.86 |
| sg17_initial | C8 | 0.228 | 210.70 |
| sg17_repeat | C1 | 0.214 | 176.17 |
| sg17_repeat | C4 | 0.224 | 196.82 |
| sg17_repeat | C8 | 0.245 | 225.45 |
| sg18_initial | C1 | 0.159 | 144.37 |
| sg18_initial | C4 | 0.189 | 165.54 |
| sg18_initial | C8 | 0.246 | 200.36 |
| sg18_repeat | C1 | 0.183 | 154.37 |
| sg18_repeat | C4 | 0.214 | 186.59 |
| sg18_repeat | C8 | 0.268 | 210.43 |

## Capability, runtime and health

Both SG17 and SG18 passed capability checks before and after long-context
validation: arithmetic, two C8 arithmetic waves, one image, four images,
structured JSON and a tool-call round trip. Both passed **128/128** concurrent
arithmetic requests and exact retrieval at **32,866, 131,170 and 299,098 input
tokens**. [SG17 receipts](../results/sg18-indexer-mhc-wo/sg17/),
[SG18 receipts](../results/sg18-indexer-mhc-wo/sg18/).

All 500 serialized server settings match the original,
with startup time and runtime observations recorded separately. Native resident
Engram, TP8/EP4/MoE-TP2, unchanged checkpoint precision, static-five speculation,
8M logical KV tokens, 128 slots, 1M context and 2,048-token prefill chunks remain
in use. [Runtime contract](../results/sg18-indexer-mhc-wo/runtime-contract.json).
The separate C1/C8 4K profiler confirmed all 192 required
kernel-dispatch checks, including unchanged expert signatures and the enabled
indexer/mHC/WO paths. Profiling is excluded from timed results.
[Dispatch receipt](../results/sg18-indexer-mhc-wo/dispatch.json).

The eight-rank guard passed throughout both suites. Lowest sampled OS-available
memory was **5.99 GiB for SG17** and
**5.32 GiB for SG18**; the floor was 2 GiB.
The final audit at `2026-09-14T11:27:30.399644+00:00` found SG18 running and idle on all
eight ranks with clean current health/transport markers and all 259 staged-file
hashes intact. The original containers and exact original cache bytes are
preserved stopped. [Final health and hash receipt](../results/sg18-indexer-mhc-wo/health.json).

## Scope and quality limits

This is the qualified v06 combined source with bounded candidate filtering,
mHC combine/norm and coefficient scheduling, and BF16 WO-A with packed WO-B
input. [Exact source and integration settings](../experiments/sg18-indexer-mhc-wo/).
The WO-A GEMM changes reduction order; unchanged weight and activation formats
do not imply bitwise parity. [Component checks](../results/sg18-indexer-mhc-wo/components/).

The prior 32K/64K/128K study measured combined-build code **86/96**, versus
original **88/96 and 89/96**, and exact answers **51/96**, versus **52/96 and
54/96**. Its 4K combined build passed code **32/32** and answers **20/32**,
matching the final original. Those quality differences remain visible; this
README capability suite does not clear broad semantic noninferiority.
The prior primary speed study showed +5.46% / +8.37% against its two controls;
the requested 30% uplift was not reached. [Full prior tables and limitations](sg18-prior-study.md).

SG18 throughput here is measured at C1/C4/C8. The historical SG11 C128 throughput
and earlier eight-million-token capacity results remain separate measurements.
This run does not measure 128 simultaneous million-token prompts or full-pool
capacity. The standard build/launcher still packages SG5; the SG18 source snapshot
describes the measured integration, not a separately tested generalized installer.
