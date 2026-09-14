# Prior long-context and 4K study

Measured 13–14 September 2026, before the separate SG18 README qualification.
The combined build is the exact v06 indexer+mHC+WO source published here.
The requested 30% primary-suite uplift was not reached. The study measured
+5.46% against the preceding original and +8.37% against the following original.
Its original report was not a promotion decision; SG18's later README result
and deployment decision are recorded separately.

This is a public extract of the completed study: numerical tables and limits
are retained, with private workspace links and obsolete live-service state
omitted. [Source identity](../experiments/sg18-indexer-mhc-wo/source-manifest.json).

## Primary performance

The primary suite uses frozen code, prose and reasoning inputs at 32K, 64K and 128K, C1/C8, one warmup and five measured 1,024-token generations per cell. Each arm has 108 runs and 486 requests. Rates count generated tokens while all requested streams are active; C8 is aggregate throughput. Prefix preparation, warmup and profiler runs are excluded.

| Reference | Overall change | C1 | C8 | Within-run 95% interval |
|---|---:|---:|---:|---|
| Preceding original | +5.46% | +3.90% | +7.04% | +0.96% to +10.02% |
| Following original | +8.37% | +6.75% | +10.02% | +2.32% to +12.37% |
| Geometric midpoint sensitivity | +6.91% | +5.32% | +8.52% | +2.31% to +10.60% |

The two original controls differ by -2.69%. Of 18 primary cells, 13 are above both controls and 0 are below both. The bootstrap resamples trials within each block; it does not quantify temporal drift. The midpoint is a sensitivity reference, not a fitted time interpolation.

The first original primary block finished at 2026-09-13T19:09:36.306547+00:00; the combined block finished at 2026-09-14T01:51:30.275139+00:00; the final original block finished at 2026-09-14T03:35:49.957880+00:00. These are separate launch blocks, not randomized per-cell alternation. Guard-start and completion timestamps are retained in the final JSON.

| Context | Content | Concurrency | Original before tok/s | Combined tok/s | Original after tok/s |
|---|---|---|---:|---:|---:|
| 32K | code | C1 | 105.93 | 106.81 | 100.93 |
| 32K | code | C8 | 309.35 | 322.88 | 304.26 |
| 64K | code | C1 | 106.50 | 106.01 | 100.40 |
| 64K | code | C8 | 304.62 | 314.70 | 298.89 |
| 128K | code | C1 | 104.52 | 107.53 | 100.79 |
| 128K | code | C8 | 295.57 | 310.28 | 287.19 |
| 32K | prose | C1 | 95.05 | 101.48 | 89.45 |
| 32K | prose | C8 | 248.44 | 304.28 | 253.57 |
| 64K | prose | C1 | 91.75 | 105.23 | 107.35 |
| 64K | prose | C8 | 327.15 | 301.99 | 299.54 |
| 128K | prose | C1 | 76.59 | 78.06 | 70.78 |
| 128K | prose | C8 | 238.04 | 295.19 | 228.27 |
| 32K | reasoning | C1 | 87.43 | 86.81 | 84.47 |
| 32K | reasoning | C8 | 259.48 | 270.61 | 249.09 |
| 64K | reasoning | C1 | 85.98 | 95.96 | 82.86 |
| 64K | reasoning | C8 | 244.93 | 261.94 | 240.08 |
| 128K | reasoning | C1 | 84.43 | 83.43 | 80.65 |
| 128K | reasoning | C8 | 234.73 | 244.67 | 231.91 |

| Work comparison | Observed verification-cycle rate | Output per cycle | Throughput |
|---|---:|---:|---:|
| Combined / first original | +3.44% | +1.95% | +5.46% |
| Combined / final original | +4.66% | +3.55% | +8.37% |
| Final / first original | -1.16% | -1.55% | -2.69% |

The work ratios multiply to reproduce throughput using the median-rate trial of each cell. These are streamed completion intervals, including scheduling and transport; they are not pure GPU latency or causal attribution to the patches. Original output-per-cycle variation remains visible in its own row.

## Factor screening and requested 4K supplement

I is bounded candidate filtering. HW combines the mHC and WO changes; IHW enables all three. The long screen has eight 32K/128K code/prose cells. The 4K supplement has six code/prose/reasoning cells. These suites have different frozen prompts and generation lengths and remain separate from the primary evaluation.

| Suite | Arm | Versus first original | Versus final original | Midpoint sensitivity |
|---|---|---:|---:|---:|
| 32K/128K screen | I | -2.26% | +6.80% | +2.17% |
| 32K/128K screen | HW | -3.95% | +4.95% | +0.40% |
| 32K/128K screen | IHW | -2.02% | +7.06% | +2.42% |
| 4K supplement | HW | -0.50% | -0.44% | -0.47% |
| 4K supplement | IHW | +6.25% | +6.31% | +6.28% |
| 4K supplement | I | +3.82% | +3.88% | Not bracketed |

The original controls differ by -8.48% overall in the long screen and -0.05% at 4K. Long-screen cell changes range from -27.09% (128K/prose/C1) to +2.09% (32K/code/C8). These differences are observed control-block variability, not a fitted drift model.

Both 4K original controls follow I because the supplement was requested while I was already running. I has no preceding 4K control. HW/IHW are temporally bracketed. Natural generation and speculative acceptance differ across runs; these four-arm results do not isolate causal kernel interactions or estimate H and W independently.

## Bounded quality

Each long-context arm produced 96 coding functions and 96 exact answers; each 4K arm produced 32 of each. Quality documents target the stated context lengths within the unchanged 512-token construction tolerance; performance inputs use their exact declared lengths. Coding passes require the frozen contract and all 30 hidden tests. The answer suite separately covers retrieval, code tracing, reasoning, structured prose and multi-document questions.

| Suite | Arm | Code first original / candidate / final original | Exact answers first original / candidate / final original |
|---|---|---|---|
| 32K/64K/128K | I | 88/96 / 88/96 / 89/96 | 52/96 / 53/96 / 54/96 |
| 32K/64K/128K | HW | 88/96 / 90/96 / 89/96 | 52/96 / 51/96 / 54/96 |
| 32K/64K/128K | IHW | 88/96 / 86/96 / 89/96 | 52/96 / 51/96 / 54/96 |
| 4K | I | 31/32 / 32/32 / 32/32 | 20/32 / 20/32 / 20/32 |
| 4K | HW | 31/32 / 32/32 / 32/32 | 20/32 / 20/32 / 20/32 |
| 4K | IHW | 31/32 / 32/32 / 32/32 | 20/32 / 20/32 / 20/32 |

| Suite | Arm | Code paired gains / regressions versus final original | Answer case-mean gains / regressions versus final original |
|---|---|---:|---:|
| 32K/64K/128K | I | 1 / 2 | 0 / 1 |
| 32K/64K/128K | HW | 3 / 2 | 0 / 3 |
| 32K/64K/128K | IHW | 1 / 4 | 1 / 4 |
| 4K | I | 0 / 0 | 0 / 0 |
| 4K | HW | 0 / 0 | 0 / 0 |
| 4K | IHW | 0 / 0 | 0 / 0 |

| Suite | Arm | Identical token sequences versus final original | Candidate repeat agreement | Final original repeat agreement |
|---|---|---:|---:|---:|
| 32K/64K/128K | I | 73/96 | 39/48 | 33/48 |
| 32K/64K/128K | HW | 71/96 | 40/48 | 33/48 |
| 32K/64K/128K | IHW | 74/96 | 41/48 | 33/48 |
| 4K | I | 28/32 | 14/16 | 14/16 |
| 4K | HW | 30/32 | 16/16 | 14/16 |
| 4K | IHW | 29/32 | 14/16 | 14/16 |

| Original-to-original control | Identical token sequences | Code paired gains / regressions | Answer case-mean gains / regressions |
|---|---:|---:|---:|
| 32K/64K/128K | 68/96 | 2 / 1 | 2 / 1 |
| 4K | 29/32 | 1 / 0 | 0 / 0 |

Every rejected coding function in these original and candidate controls used a prohibited import in the topological-ordering contract. All seven other coding contracts passed every repetition. The fixed no-import rule remains part of the score; identifying this failure mode does not waive it or clear the candidate regression flag.

Original repeats can change, but that does not erase candidate regressions. Both 4K original quality controls follow I. These synthetic tasks do not establish broad semantic noninferiority or quality throughout the configured one-million-token capacity. Log-probabilities are compared only at common-conditioning positions, including the first divergent decision; top-20 overlap is not a full-distribution distance. All individual comparisons and repeatability statistics are retained in the linked JSON evidence.

## Implementation and interpretation

- Bounded indexer filtering retains the original full-width top-k selector, candidate budget, tie behavior and newest-block rule. Graph-owned state clears dirty tails and prior publication bits; boundary, shrink, poisoned-tail and reuse cases passed exact comparison.
- The SM121 mHC combine/norm path retains intermediate BF16 rounding and the original C1 path. The coefficient projection uses the qualified one-stage schedule without changing its arithmetic or reduction order.
- TP8 WO-A uses the qualified small-row BF16 path and explicit packed WO-B input. The actual receiving quantizer and linear consumer were tested. The changed GEMM reduction order remains a numerical/model-quality consideration.
- Expert source and captured kernel signatures are controlled. The experimental cache union preserves existing tactic keys and values; its original per-rank file bytes were restored at handback.
- v07 shared-expert contraction padding has component and actual linear-consumer evidence but was not part of the I/HW/IHW serving screen. The R37 dependency changes were inspected only. Neither is credited with serving gains here.

The first final-original 128K/C1 trace spans 51.56 ms and shows a rank-six slowdown in unchanged sparse-attention and other kernels, with longer collective spans on other ranks. A repeat after all timed and quality controls spans 39.20 ms at C1 and 118.48 ms at C8. Both traces retain identical captured dispatch; all per-rank sparse durations are in the final JSON. The cause of the first anomaly remains unidentified, and neither diagnostic replaces or changes the unprofiled throughput measurements.

Large component speedups did not establish a comparable engine-level benefit. The original score producer was already bounded; the new filtering replaces a smaller region. Disjoint timeline occupancy and the observed output-per-cycle decomposition are diagnostics, not causal ceilings or throughput forecasts. The unexplained first HW rank-four sparse-attention outlier is retained; a later repeat did not reproduce it.

The same native weights, precision, static-five DSpark, TP8/EP4/MoE-TP2 topology, transport, one-million context capacity, eight-million logical KV tokens and 128 request slots were used. All collected performance comparisons pass independent token/work accounting and the eight-rank 2 GiB reserve guards.
