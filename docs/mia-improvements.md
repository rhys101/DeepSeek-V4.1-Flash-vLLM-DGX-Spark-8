# Applying Mia's work to eight Sparks

The live eight-Spark vLLM deployment now uses **Mia-inspired NCCL buffer/channel settings and guarded small-M FlashInfer b12x MXFP8 routing**. The selected profile passed text, vision, prefill, graph and C1–C8 serving checks and was promoted on all eight nodes on 11 September 2026. The previous launch configuration remains available on the original cluster for rollback.

## What Mia demonstrated

Review pinned to [MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks at e59e6eb](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks/tree/e59e6eb67479aa68f6fa700c600dc90a0729b5ec). Their measured system is a three-Spark **SGLang** deployment. Its README reports 37.9 tok/s single-stream prose and 78.6 tok/s aggregate prose with four streams. Their four-Spark profile is explicitly configuration/script-path validated only, not boot-tested at that revision; those performance numbers should not be presented as four-Spark measurements.

Two findings transfer directly. Small-M dense MXFP8 projections benefit from FlashInfer's SM12x b12x implementation; Mia attributes a reduction from about 50–52 ms to 17 ms per speculative step to these projections. NCCL's connection buffers also pin substantial unified RAM, so reducing buffer sizes/channels releases memory. Our deployment used 64 communication channels and two communicators, making the latter especially relevant. Mia's timings use a different engine, rank count and workload and are not a matched speed comparison with this repository.

## Changes applied

| Mia technique | Eight-Spark adaptation |
|---|---|
| Reduced NCCL connection buffers | Set `NCCL_BUFFSIZE=1048576`, `NCCL_LL128_BUFFSIZE=262144`, `NCCL_PROTO=^LL128`, `NCCL_MAX_NCHANNELS=8`; test both collective correctness and serving throughput |
| Small-M b12x dense MXFP8 | Original vLLM routing patch: b12x for SM12x, 0 < M ≤ 128, K divisible by 128 and BF16/FP16 output; CUTLASS for other shapes |
| TP3 head/group/expert padding | Not needed for TP8; existing dimensions divide across eight ranks |
| NVMe Engram staging | Keep our working native resident Engram; eight nodes already provide sufficient capacity |
| SGLang allocator/prefill hooks and MoE finalize fixes | Engine-specific changes; the current vLLM allocator, chunked prefill and DeepGEMM MXFP4 path passed our checks |

The [runtime profile](../configs/profiles/dspark-300k.json) contains all selected settings. Context remains 300,000, batch tokens 8,192, request slots eight, memory utilization 0.80, DSpark k=5, vision up to four images and exact target/draft graph sizes. The model snapshot and quantization formats are unchanged. [The original vLLM patch](../patches/small-m-mxfp8.patch) uses upstream FlashInfer kernels; it does not copy Mia's SGLang adapter. Attribution and licenses are in [NOTICE](../NOTICE).

`NCCL_MAX_NCHANNELS` is a deprecated but supported control in the tested stack; retain the measured spelling for reproduction. Newer NCCL documentation describes `NCCL_MAX_CTAS` as its replacement. Buffer/channel settings are hardware- and workload-dependent. [NVIDIA NCCL environment documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html).

## Measured serving result

| Measurement | Previous eight-Spark run | Current profile | Change |
|---|---|---|---|
| C1 mean decode tok/s | 68.33 | 68.41 | +0.1% |
| C6 mean aggregate tok/s | 170.81 | 205.35 | +20.2% |
| C6 mean decode tok/s | 33.78 | 39.68 | +17.5% |
| C8 mean aggregate tok/s | 187.80 | 237.27 | +26.3% |
| C8 mean decode tok/s | 27.53 | 35.08 | +27.4% |
| 93,335-token prefill TTFT, s | 36.38 | 33.51 | -7.9% |

Category means exclude counting. Aggregate rate includes prefill; decode rate excludes TTFT. The single-stream mean is essentially flat while concurrency improves. These are measured deployment changes, not a prediction of a proportional speedup for every request.

The baseline is the archived 10 September run. An unchanged baseline restart on 11 September failed in rank 7's post-capture Triton sampler warmup with `CUDA operation not permitted`, before any new baseline measurements. Both comparison candidates then completed startup and validation, but this does not establish the cause of that startup failure. The before/after results therefore include day/process variation.

The same-day comparison separates the selected communication settings from the added kernel routing:

| Measurement | NCCL settings only | NCCL + MXFP8 routing | Additional change |
|---|---|---|---|
| C1 mean aggregate tok/s | 59.00 | 60.84 | +3.1% |
| C1 mean decode tok/s | 66.04 | 68.41 | +3.6% |
| C6 mean aggregate tok/s | 201.43 | 205.35 | +1.9% |
| C6 mean decode tok/s | 39.34 | 39.68 | +0.9% |

Most of the improvement is already present with the NCCL settings. The small additional MXFP8 gain is within plausible run-to-run variation; there is one batch per category/concurrency and no repeated-run confidence interval. Individual categories can be slower with the added route: C6 coding aggregate measured 312.14 tok/s with NCCL alone and 305.24 with both changes. The combined profile was retained after passing the complete C1–C6, C8, vision and prefill checks, not on a claim that every category improves.

## Kernel and collective evidence

The dense-kernel probe tested 32 combinations across four projection shapes, M=1 through 1,024, both backends and changed-input/scales graph replay against an independent dequantized FP32 reference. All passed the 0.008 relative L2 limit. Several M=5/6 projection calls were about 10–13× faster with b12x than the direct CUTLASS default in this isolated probe; replicated Engram projections were about 2.5–2.8× faster. Those ratios compare isolated kernel defaults. The real vLLM path autotunes its kernels and includes MoE, attention, speculation and communication, so they do not predict serving speedups. The installed integration and boundary M=128/129 were also checked, and the full model's autotune records showed b12x runners for the actual eligible projections.

Eight-rank collective probes used two independent communicators, correctness assertions and CUDA graph timing. The table averages each rank/communicator's median latency, in microseconds:

| Message bytes | Original µs | Selected 8-channel µs | 16-channel µs |
|---|---|---|---|
| 20,480 | 136.5 | 137.7 | 135.2 |
| 122,880 | 166.3 | 168.4 | 169.6 |
| 983,040 | 1366.8 | 360.2 | 553.2 |
| 16,777,216 | 2003.2 | 1336.9 | 1331.4 |
| 67,108,864 | 5015.4 | 5014.4 | 5031.8 |

Small-message latency remained close, the approximately 1 MB and 16 MB cases improved, and the 64 MiB case stayed essentially unchanged. Eight channels provided the lowest shared-memory footprint and better approximately 1 MB results than sixteen, so eight was selected. These probes report Torch NCCL 2.29.7; vLLM separately logs loaded NCCL 2.30.7+cuda13.3. Full serving results above are the direct performance evidence for the deployed library path.

Mean Linux shared memory in the controlled probe fell from **11.61 GiB to 0.425 GiB per node**, approximately **11.18 GiB recovered**. This is an OS shared-memory observation, not a direct NCCL allocator measurement. The final serving snapshot showed **13.86–15.78 GiB available RAM per Spark**, with about 0.40 GiB shared memory. Two-second benchmark samples stayed above 14.05 GiB available; a reserve is not guaranteed for untested workloads. [Memory details](memory-footprint.md).

## Validation and limits

The final profile completed 54 main batches / 189 requests, four cold-prefill cases to 93,335 input tokens, and nine C8 batches / 72 requests. Arithmetic, image OCR/colors/shapes and four-image order passed. All eight live containers had the same image/source hash, zero restarts, no OOM kills and no logged model traceback. The measured code and environment are now the canonical launch files on the cluster.

All 189 category input counts and four prefill sizes match the pinned community reference; 86 output lengths differ. This is not a broad quality evaluation, a full 300K request test, a verified 1M deployment or a controlled eight-versus-three-node SGLang comparison. The full combined Dockerfile has not had a separate clean rebuild; the measured image was assembled from the validated base plus the checked MXFP8 source layer.

[Current raw results](../results/2026-09-11/) · [Comparison inputs](../results/reference/) · [Validation commands](validation.md) · [Build/source pins](build-and-pins.md)

## Remaining experiments

The next code candidate is `wo_a`, the grouped attention output projection. Read-only inspection of the running image confirmed that GB10 falls back to BF16: the MXFP8 BMM selector admits DeepGEMM only on SM100-family devices, and the output-projection helper uses `torch.bmm` for BF16 weights. Our TP8 configuration has eight global output groups and one local group per rank, reducing each local projection to a regular M×4096 by 4096×1024 matrix multiplication. That may allow reuse of FlashInfer's existing b12x GEMM. It is a proposed experiment, not a measured gain: it also introduces FP8 activation quantization where the current fallback keeps BF16 activations, so kernel, graph, logits and model-output checks are necessary. The fallback already dequantizes weights once at loading; there is no per-step weight-dequantization saving to claim.

A short all-rank profile at C1 and C8 should first establish the time spent in this projection, MoE, communication and sampling on the optimized vLLM deployment. Mia's SGLang profile cannot supply those percentages for TP8. The speculative window is another targeted experiment: compare k=3, 4 and 5 with matching target/draft graph sizes, accepted tokens per step and both aggregate and individual-stream throughput. Verifying five drafted tokens is not necessarily optimal at every concurrency. During the C8 controller window (including three C1 warmups), counters recorded 8,222 accepted tokens out of 15,985 drafted (51.4%), averaging 2.57 accepted tokens per five-token draft. The fifth position was accepted on 33.3% of drafts. That is evidence to run the comparison, not proof that shortening the window helps. [Counter deltas](../results/2026-09-11/speculation-c8-window.json).

For aggregate-throughput work, two independent TP4 replicas on the eight-node fleet are a separate architectural experiment. They shorten each tensor-parallel collective but duplicate model state and would require revisiting Engram disk/offload placement; the current resident-memory TP8 layout cannot simply be divided into two. Neither speed nor quality has been measured for that arrangement here. Existing inverse-RoPE/quantization and qnorm/RoPE fusion, local weight placement and load-time weight dequantization should be accounted for before proposing duplicate optimizations.
