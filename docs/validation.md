# Validation and known limits

The current deployment was validated on all eight Sparks on 11 September 2026. Image identity and source hashes are recorded in [versions.lock.json](../versions.lock.json) and [the current evidence](../results/2026-09-11/).

- All eight nodes run the identical patched image. The selected NCCL environment, M=128 kernel threshold, canonical launch files and live process state were checked after promotion. No OOM kills, restarts or logged model tracebacks were found.
- Dense MXFP8: 32 shape combinations, two backends, eager output and changed-input/scales CUDA graph replay passed an independent dequantized FP32 reference. The relative L2 acceptance limit was 0.008. Shapes cover M=1, 5, 6, 40, 48, 128, 256 and 1,024 across four model projection dimensions.
- The installed vLLM integration selected b12x at M=1, 5, 6, 40, 48 and 128; CUTLASS at M=129 and 256. Bias, a three-dimensional input and graph replay with changed inputs passed the checked CUTLASS reference (`rtol=0.01`, `atol=0.0625`).
- Eight-rank, two-communicator all-reduce/all-gather correctness passed with original, selected eight-channel and alternative sixteen-channel settings. Timings cover 20 KB through 64 MiB with CUDA graph replay. Torch reported NCCL 2.29.7; the full serving process logged loaded NCCL 2.30.7+cuda13.3. Model benchmarks provide the direct serving evidence.
- The full model loaded and captured target/draft graphs. Arithmetic returned 323. Single-image OCR, colors and shapes passed; a four-image request returned all codes in order.
- The unchanged community suite completed 54 batches / 189 requests at C1–C6, four cold-prefill cases through 93,335 tokens, and a separate C8 run of nine batches / 72 requests. All main-run input counts match the reference. The benchmark sampled available memory every two seconds on all nodes.
- Eleven local configuration/command-construction tests pass. The generic packaged recipe has not had a separate complete cold rebuild or clean-cluster installation.

The unchanged underlying stack has dated [base-stack validation](../results/base-stack-validation/) from 10 September: six fused qnorm/RoPE/KV-insert cases; sparse metadata/indexer checks; 24 sparse attention/changed-query graph cases; backend prefill at 80, 1,024 and 8,192 queries with an independent FP32 reference; and nine MHC kernel numerical checks. Prefill used the upstream 0.05 absolute/relative tolerance after a tighter preliminary check failed. Those checks describe the retained attention/native kernels, while the new MXFP8 route and current full model were checked separately above.

Before loading the full model, run GPU checks on an idle Spark using your newly built image (replace `IMAGE`):

```bash
mkdir -p .local/gpu-checks
for check in check-native.py check-runtime.py check-spec-graphs.py check-prefill.py; do
  docker run --rm --gpus all --ipc host -e VLLM_PLUGINS= -e MAX_JOBS=2 \
    -v "$PWD/runtime:/kit:ro" --entrypoint python3 IMAGE "/kit/$check"
done
for check in check-mxfp8.py check-mxfp8-routing.py; do
  docker run --rm --gpus all --ipc host -e VLLM_PLUGINS= -e MAX_JOBS=2 \
    -v "$PWD/runtime:/kit:ro" -v "$PWD/.local/gpu-checks:/checks" \
    --entrypoint python3 IMAGE "/kit/$check" --out "/checks/$check.json"
done
```

The Dockerfile includes the small upstream test subset required by `check-native.py`. The originally assembled image used a separately mounted copy of the identical upstream tests. Run the cluster's 2/4/8-rank NCCL smoke commands before serving; `runtime/bench-nccl.py` additionally provides the two-communicator timing probe for use under an eight-node `torchrun` launch with the selected fabric/environment, host IPC, RDMA device, unlimited memlock and `nofile=1048576:1048576`.

After launch, run actual API checks:

```bash
python3 scripts/cluster.py smoke configs/cluster.local.json
python3 runtime/smoke-vision.py --base http://127.0.0.1:8000/v1 --out .local/vision-check-1
```

Use a fresh output directory. Fixtures are included in `runtime/vision-fixtures`. Run `python3 -m unittest discover -s tests -v` for local configuration checks; CI does not claim GPU validation.

These tests do not establish broad model quality, tool-use accuracy, long-context answer quality or successful full 300K requests. The before/after throughput baseline is the archived 10 September result; see [comparison limits](mia-improvements.md).

## Quality before further speed tuning

The checks above do not establish broad model-quality parity. Evaluate task accuracy, reasoning, tool reliability and relevant vision/long-context capability before accepting precision reductions or other capability tradeoffs. The current numerical and smoke results alone are insufficient to justify the proposed BF16-to-FP8 activation change in `wo_a`. [Speed versus model capability](speed-and-quality.md).
