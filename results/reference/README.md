# Comparison inputs

These JSON files retain their original bytes and run identifiers.

- `bench-tony-boot10.json`: Tony's four-Spark boot 10 at commit `ca662ac35193c69ace9cee37f13a94abf2eff0fc`.
- `bench-spark8-before.json` and `bench-spark8-before-c8.json`: the previous eight-Spark profile, measured on 10 September 2026 with image `sha256:4514e1e972669763b1ceb94b6f15f70e66e6ee8f381a502246217f165b63a342`.
- `bench-spark8-nccl-only.json`: 11 September comparison on that same image with the selected NCCL settings; C1 and C6 plus the four prefill cases. The small-M source patch was not active.

The unchanged previous profile was restarted to obtain a fresh baseline but failed in rank 7's post-capture Triton sampler warmup (`CUDA operation not permitted`). No new baseline benchmark was completed, so the before/after comparison uses the dated archived baseline. The NCCL-only and combined profiles both booted and completed their checks; this does not prove which setting caused the earlier startup failure. Neither comparison has repeated-run confidence intervals.

[Analysis and attribution](../../docs/mia-improvements.md)
