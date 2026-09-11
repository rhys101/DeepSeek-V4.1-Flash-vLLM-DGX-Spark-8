# Recorded deployment results

The current eight-Spark profile uses reduced NCCL buffers/channels and small-M MXFP8 b12x routing, with native resident Engram, vision, DSpark k=5, CUDA graphs, a 300K context cap and eight request slots.

| Directory | Contents |
|---|---|
| [2026-09-11](2026-09-11/) | Current C1–C6, C8, prefill, image/text checks, memory samples, image identity, MXFP8 and collective checks |
| [reference](reference/) | Unchanged inputs used for the Tony comparison and the measured optimization comparison |
| [base-stack-validation](base-stack-validation/) | Dated checks for the unchanged underlying kernels, packages and model snapshot |

Raw run identifiers are preserved. [Interpretation](../docs/comparison.md) · [Mia adaptations](../docs/mia-improvements.md) · [Method](../docs/benchmark-method.md)
