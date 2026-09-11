# Eight-Spark deployment — 11 September 2026

The main community run contains 54 batches / 189 requests and four cold-prefill cases; the separate C8 run contains nine batches / 72 requests. Arithmetic, single-image OCR/colors/shapes and four-image ordering passed. All eight nodes use the same patched image and remained running without OOM kills, restarts or logged model tracebacks.

The manifest records file hashes and input-count comparisons. `health.json` includes loaded-NCCL and graph/KV startup evidence, actual runtime environment, and the final checks. `memory-samples.jsonl` contains two-second `/proc/meminfo` observations; the summary separates the benchmark window from loading/warmup.

`mxfp8-validation.json` covers 32 shapes and two backends with changed-input/scales graph replay and an independent FP32 reference. `mxfp8-routing-validation.json` checks the installed vLLM integration and its M=128/129 boundary. The three NCCL files contain model-unloaded, eight-rank, two-communicator correctness and timing probes. Their Torch-reported NCCL is 2.29.7; serving logs separately identify vLLM's loaded NCCL as 2.30.7+cuda13.3. Full-model measurements are the evidence for serving performance.

These are bounded validation and throughput measurements, not a broad model-quality or full 300K-context evaluation. See [validation](../../docs/validation.md) and [method](../../docs/benchmark-method.md).
