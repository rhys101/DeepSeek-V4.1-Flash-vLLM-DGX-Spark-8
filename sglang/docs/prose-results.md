# EP4 / SG5: two-way tensor parallelism within each expert group

Validated 11 September 2026, 20:06–20:16 UTC. This is the separate sparkDash prose benchmark; the main release report uses the matched community coding workload for vLLM comparison. TP8 remains global, with four expert groups and two tensor shards per expert. Target and all three draft stages use FlashInfer CUTLASS MXFP4/MXFP8. Draft length remains five.

## Speed

| Concurrent requests | Two trials, aggregate decode tok/s | Mean | Change from SG3 |
|---|---:|---:|---:|
| 1 | 67.55 / 67.05 | 67.300 | +5.12% |
| 2 | 109.66 / 110.49 | 110.075 | +9.22% |
| 3 | 142.68 / 142.83 | 142.755 | +9.09% |
| 4 | 154.66 / 154.13 | 154.395 | +3.77% |
| 6 | 211.40 / 210.88 | 211.140 | +19.03% |
| 8 | 227.98 / 229.30 | 228.640 | -2.26% |

These are sparkDash prose results, not the vLLM community coding workload. C1/C4 are 1.4473× / 1.4817× Mia's reported four-Spark figures and remain below 1.5×.

[Raw trials](../results/ep4/sparkdash/result.json) · [Summary](../results/ep4/sparkdash-summary.json) · [Earlier SG3 reference](sg3-reference.md).

Prose gains over SG3 were +5.12% at C1, +3.77% at C4 and +19.03% at C6; C8 declined 2.26%. The two greedy C1 previews match (1,073 characters), but previews are not complete token-identity evidence. The two trials are observations without confidence intervals.
