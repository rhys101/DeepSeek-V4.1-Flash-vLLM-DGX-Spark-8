# Validation and known limits

This serving configuration was validated on the original eight-Spark deployment with image ID recorded in [versions.lock.json](../versions.lock.json):

- The identical image was checked on every node; an eight-rank all-reduce/all-gather test passed over IB/RoCE before loading.
- Six unchanged upstream fused DeepSeek qnorm/RoPE/KV-insert cases passed for 4, 48 and 2,048 tokens, including TP8 heads and 64-state pages.
- Sparse metadata, 64-state indexer scheduling and per-row top-k against PyTorch passed. Vision reserves the 1,152-wide selection.
- Twenty-four sparse-attention cases cover DSpark/target batches of 5, 6, 40 and 48 queries, text and vision widths, and supported page combinations. Three graph replays per case with changed queries are finite and bitwise equal to eager output.
- Actual backend prefill at 80, 1,024 and 8,192 queries was finite and passed an independent FP32 reference on the first 16 rows, using the upstream FlashInfer DeepSeek tolerance of 0.05 absolute and relative. Maximum absolute differences were about 0.01086, 0.00422 and 0.00368 respectively. The harness asserts that both compression ratios select the actual 64-state layout. These checks do not establish correctness at a tighter tolerance; a tighter preliminary check failed. Coverage uses the page layouts selected by the actual backend.
- Nine MHC kernel numerical cases passed against references. Full target and draft graph capture completed, reporting 0.65 GiB graph memory on rank 0.
- Arithmetic returned 323. Single-image OCR, colors and shapes passed; a four-image request returned every code in order. The schema check uses a prompt that explicitly requests string values.
- The unchanged category benchmark completed 54 batches / 189 measured requests, then four cold-prefill cases at 2,950, 11,592, 46,810 and 93,335 input tokens. Every input count matched boot 10. No full 300K request or long-context answer-quality test was run.
- The same serving configuration later completed a separate C8 extension: nine batches / 72 measured requests, with no repeated prefill sweep; all eight nodes remained healthy. The [memory audit](memory-footprint.md) reports the physical footprint and limited OS-available RAM.
- After the benchmark, every container was running on the expected image/configuration with zero restarts, no OOM kills and no logged model traceback.
- The original GPU-state probe passed separately on all eight GPUs with the model unloaded. It did not reproduce a large collapse during those measurements. Normal clock samples during serving do not establish that no hidden state transition can occur under load.
- The generic configuration, staging and launch-preview commands ran on all eight original nodes without loading a second model. Eleven local configuration/command-construction tests pass. The packaged combined cold build and clean-cluster installation have not been reproduced separately.

Raw vision, package, source, health and benchmark evidence is in [the results directory](../results/2026-09-10/).

Before loading the full model, run the packaged GPU checks on an idle build head using your newly built image (replace `IMAGE`):

```bash
for check in check-native.py check-runtime.py check-spec-graphs.py check-prefill.py; do
  docker run --rm --gpus all --ipc host -e VLLM_PLUGINS= -e MAX_JOBS=2 \
    -v "$PWD/runtime:/kit:ro" --entrypoint python3 IMAGE "/kit/$check"
done
```

The Dockerfile includes the small upstream test subset required by `check-native.py`. The originally assembled image used a separately mounted copy of the identical upstream tests.

After launch, run actual API checks:

```bash
python3 scripts/cluster.py smoke configs/cluster.local.json
python3 runtime/smoke-vision.py --base http://127.0.0.1:8000/v1 --out .local/vision-check-1
```

Use a fresh output directory. The fixture images are included in `runtime/vision-fixtures`. Run `python3 -m unittest discover -s tests -v` for local configuration checks; CI does not claim GPU validation.

These smoke checks and throughput runs are not a broad model-quality, tool-use or reasoning evaluation.
