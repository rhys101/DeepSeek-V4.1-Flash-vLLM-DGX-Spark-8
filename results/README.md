# Recorded deployment results

All eight-Spark evidence here uses the documented FlashInfer 0.7.0rc1 serving configuration, with native resident Engram, vision, DSpark k=5, CUDA graphs, 300K context cap and eight request slots.

| Directory | Contents |
|---|---|
| [2026-09-10](2026-09-10/) | C1–C6 and cold prefill; pinned Tony comparison; image, kernel, source and health evidence |
| [2026-09-10-c8](2026-09-10-c8/) | Separate C8 measurement on the same server |
| [2026-09-10-memory](2026-09-10-memory/) | Per-node memory snapshot |

Raw run identifiers and measurements are preserved for provenance. [Interpretation](../docs/comparison.md) · [Method](../docs/benchmark-method.md)
