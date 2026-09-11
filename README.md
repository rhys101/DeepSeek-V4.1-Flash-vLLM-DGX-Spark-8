# DeepSeek V4.1 Flash on eight DGX Sparks

Serve `deepseek-ai/DeepSeek-V4.1-Flash` across **eight NVIDIA DGX Sparks** with **TP8, RAM-resident Engram, DSpark k=5, CUDA graphs and vision**.

This repository contains two measured serving deployments recorded on **11 September 2026**:

| Engine | Parallelism | Build, configuration and evidence |
|---|---|---|
| **SGLang EP4 (SG5)** | TP8 / EP4 / MoE-TP2 | [SGLang release](sglang/README.md) |
| **vLLM** | TP8 | The vLLM recipe and results below |

Both use the pinned DeepSeek V4.1 Flash checkpoint, resident Engram, five-token DSpark drafting, eight request slots, four-image support and a 300,000-token context cap. Their software versions, expert partitioning and memory accounting differ.

## SGLang EP4 versus vLLM

**111.29 tok/s single-stream coding decode**, versus **95.91** on the existing vLLM deployment (16.0% higher).

| Measurement (tok/s) | vLLM, eight Sparks | SGLang EP4, eight Sparks | EP4 change |
|---|---|---|---|
| Coding C1, decode per stream | 95.91 | 111.29 | +16.0% |
| Coding C4, aggregate | 239.60 | 274.07 | +14.4% |
| Coding C6, aggregate | 305.24 | 390.93 | +28.1% |
| Coding C8, aggregate | 341.44 | 431.82 | +26.5% |
| Category mean C4, aggregate | 157.35 | 192.27 | +22.2% |
| Category mean C6, aggregate | 205.35 | 255.66 | +24.5% |
| Category mean C8, aggregate | 237.27 | 291.33 | +22.8% |

[Complete comparison, token counts and limits](sglang/docs/community-comparison.md).

The comparison uses the same unchanged community benchmark and coding prompt. Single-stream decode excludes time before the first token; aggregate throughput includes prefill and batch wall time. The full [SGLang release](sglang/README.md) includes source pins, build/launch scripts, numerical checks, capability checks, long-context retrieval and raw measurements. Its separate sparkDash prose benchmark is documented independently.

## vLLM results

The vLLM deployment combines [Tony's pinned vLLM/FlashInfer recipe](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark/tree/ca662ac35193c69ace9cee37f13a94abf2eff0fc) with [Mia-inspired NCCL settings and small-M MXFP8 routing](docs/mia-improvements.md). Unless a section links explicitly to `sglang/`, the remaining instructions and results describe vLLM.

**95.9 tok/s single-stream coding decode.** At six concurrent requests, coding aggregate throughput is **305.24 tok/s** and the eight-category mean is **205.35 tok/s**. Eight concurrent requests reaches **237.27 tok/s** category mean.

| Measurement (tok/s) | Tony, 4 Sparks | This deployment, 8 Sparks | Ratio |
|---|---|---|---|
| Coding C1, decode per stream | 73.78 | 95.91 | 1.30× |
| Coding C4, aggregate | 123.80 | 239.60 | 1.94× |
| Coding C6, aggregate | 225.48 | 305.24 | 1.35× |
| Category mean C4, aggregate | 85.72 | 157.35 | 1.84× |
| Category mean C6, aggregate | 131.86 | 205.35 | 1.56× |

The category mean covers code, JSON, math, reasoning, tables, summary, prose and narrative; counting is excluded. C means concurrent requests. Aggregate throughput includes prefill; per-stream decode excludes time before the first token.

C8 adds 15.5% mean aggregate throughput over C6 while reducing mean per-stream decode by 11.6% (35.08 versus 39.68 tok/s). Use C6 when individual response speed matters and C8 when aggregate throughput matters more. The server has eight request slots; Tony's pinned reference stops at C6.

Compared with the previous eight-Spark measurements, aggregate throughput rose **20.2% at C6** and **26.3% at C8**. Single-stream mean decode is essentially unchanged. The same-day NCCL-only comparison accounts for most of the improvement; the additional MXFP8 routing gain is modest and not established with repeated-run confidence intervals. See [the measured adaptation results](docs/mia-improvements.md).

### Prose on three, four and eight Sparks

| Prose measurement (tok/s) | Mia: 3 Sparks, SGLang | Tony: 4 Sparks, vLLM | This deployment: 8 Sparks, vLLM |
|---|---|---|---|
| C1 decode per stream | 37.9 | 24.37 | 41.28 |
| C4 decode per stream | 20.9 | 18.37 | 26.52 |
| C4 aggregate | 78.6 | 69.94 | 93.45 |

Mia's [reported prose results](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks/blob/e59e6eb67479aa68f6fa700c600dc90a0729b5ec/README.md) come from **three Sparks running SGLang with NVMe Engram**. Tony and this deployment use the same community suite; Mia's prompt, output budget and measurement details have not been established to match it. This is a contextual prose comparison, with no hardware-scaling ratios inferred. Mia's four-Spark profile was configuration-checked but not boot-tested at the reviewed revision. Full C1–C4 decode, aggregate and TTFT figures are in [the comparison](docs/comparison.md#prose-comparison-with-mias-three-sparks).

Tony's repository was checked through `458fade` on 11 September 2026. His community benchmark remains byte-for-byte unchanged; the new commits add vision/tool checks and restore tooling. [Upstream review](docs/upstream-status.md).

### Cold prefill

Cold, unique prompts with a one-token reply. Actual input sizes match Tony's four-Spark reference; rates are prompt tokens divided by time to first token (TTFT).

| Actual prompt tokens | Tony TTFT | This deployment TTFT | Tony prefill tok/s | This deployment prefill tok/s |
|---|---|---|---|---|
| 2,950 | 3.270 s | 1.059 s | 902.2 | 2,785.5 |
| 11,592 | 11.293 s | 4.008 s | 1,026.5 | 2,892.5 |
| 46,810 | 30.426 s | 15.844 s | 1,538.5 | 2,954.3 |
| 93,335 | 78.173 s | 33.505 s | 1,194.0 | 2,785.7 |

The main run completed 54 category batches / 189 requests and these four prefill cases. C8 was measured separately with nine batches / 72 requests on the same configuration. All 189 category input counts match the reference; 86 output lengths differ. These are single measurements without repeated-run confidence intervals. A 300K context cap is configured; the longest measured prompt was 93,335 tokens, with no full 300K or long-context quality evaluation.

[Comparison and concurrency](docs/comparison.md) · [Complete tables](docs/benchmark-tables.md) · [Method](docs/benchmark-method.md) · [Raw results](results/)

## Speed versus model capability

**A throughput gain is not worthwhile if it causes a disproportionate loss of model capability.** Compare answer quality, reasoning, tool reliability and useful context alongside speed.

The deployed NCCL tuning and dense-kernel routing retain the existing model and quantization formats. They passed numerical and functional checks, but a broad quality evaluation has not been completed. Proposed lower-precision `wo_a` activations or further weight/KV quantization need separate quality evidence before being treated as improvements.

The headline benchmarks run with **thinking off** and fixed short output budgets. That mode can trade reasoning capability for less work; these numbers do not describe full reasoning-mode performance. Reducing context, truncating answers or disabling vision also changes the service's capabilities. [Quality-sensitive options and evaluation criteria](docs/speed-and-quality.md).

## Serving configuration

| Setting | Value |
|---|---|
| Hardware | Eight GB10 DGX Sparks, 128 GB shared CPU/GPU memory each |
| Parallelism | TP8 / PP1 / DP1 |
| Engram | Native, resident in memory; `cpu_offload=false` |
| Context cap / request slots | 300,000 tokens / 8 |
| Prefill batch / memory utilization | 8,192 tokens / 0.80 |
| Speculation | Native DSpark k=5 |
| CUDA graphs | Target and draft; FULL_AND_PIECEWISE |
| Prefix caching | Enabled |
| Vision | Enabled, up to four images per request |
| NCCL buffers / channels | 1 MiB Simple, 256 KiB LL128, `^LL128`, 8 channels |
| Dense MXFP8 | FlashInfer b12x for eligible SM12x shapes with M ≤ 128; CUTLASS otherwise |
| API model alias | `deepseek-v41-flash` |

The [profile](configs/profiles/dspark-300k.json) records the environment and flags; the runner derives the exact target/draft graph sizes. The comparison uses twice Tony's hardware: C6 category throughput is 1.56× his result, giving lower throughput per Spark. TP8/TP4, Engram placement, dense kernel routing and fabric differ; this does not isolate RAM-versus-disk performance.

All eight containers passed health checks with no OOM kills or restarts. Arithmetic, image OCR/colors/shapes and four-image ordering passed. [Validation](docs/validation.md) covers the added kernels and full serving profile as well as dated evidence for the unchanged base stack.

Linux reported **13.86–15.78 GiB available RAM per Spark** at the final snapshot, with about 0.40 GiB shared memory. During the benchmark, two-second samples stayed above 14.05 GiB available on every node. These observations do not guarantee a reserve for all workloads. See [memory accounting](docs/memory-footprint.md).

## Requirements

Linux ARM64 DGX Sparks; Docker with NVIDIA GPU support and Buildx; Python 3.10+, Git, SSH and rsync; a working RDMA fabric with MTU 9000. The checkpoint must be available locally on every node: approximately 510 GB of tensor data per copy, plus space for build caches and images. See [hardware and networking](docs/hardware-and-network.md).

Source pins and observed package versions are in [versions.lock.json](versions.lock.json). They are not a hermetic package lock. The running image was assembled from validated build stages; the packaged combined wrapper and clean-cluster installation have not had a separate complete reproduction. No public prebuilt image is provided.

## Quick start

Perform cluster operations on rank 0. All inter-node SSH and rsync operations bind to its configured fabric address. Confirm SSH host keys and routes before beginning.

1. Build on an idle Spark:

   ```bash
   bash scripts/build.sh
   docker image inspect --format '{{.Id}}' deepseek-v41-spark8:2026-09-11-mxfp8
   ```

   The build uses the pinned ARM64 nightly base, rebuilds the stable CUDA extension, overlays pinned feature source, applies the reference attention and small-M MXFP8 patches with zero fuzz and hash checks, installs FlashInfer 0.7, and compiles its two SM12x modules. Details: [build and pins](docs/build-and-pins.md).

2. Obtain the model at revision `df42c109f1defefcbfcedbe7d905718a12266e40` using your usual Hugging Face download workflow. The directory must contain `config.json`, the tokenizer files, the index, and all 48 safetensors shards. Credentials and weights stay outside this repository.

3. Copy and edit the cluster configuration:

   ```bash
   cp configs/cluster.example.json configs/cluster.local.json
   python3 scripts/cluster.py validate configs/cluster.local.json
   ```

   Set all eight fabric IPs, login users, HCA names, interfaces, model paths, deployment directory, and the **image ID from your own build**. The example uses documentation-only IP addresses. The serving profile is `profiles/dspark-300k.json`.

4. Distribute from rank 0 if the image or checkpoint is not already present on every node:

   ```bash
   python3 scripts/distribute.py image configs/cluster.local.json
   python3 scripts/distribute.py model configs/cluster.local.json
   ```

   Image transfers verify SHA-256 and imported image IDs. Model rsync uses checksums, follows snapshot symlinks, resumes partial copies, and does not delete destination files.

5. Stage the node configurations and run checks:

   ```bash
   python3 scripts/cluster.py configure configs/cluster.local.json
   python3 scripts/cluster.py preflight configs/cluster.local.json
   python3 scripts/cluster.py validate-config configs/cluster.local.json
   python3 scripts/cluster.py nccl configs/cluster.local.json --nodes 2
   python3 scripts/cluster.py nccl configs/cluster.local.json --nodes 4
   python3 scripts/cluster.py nccl configs/cluster.local.json --nodes 8
   ```

   Run kernel validation before the full model is loaded; commands are in [validation](docs/validation.md). Preflight checks availability and checkpoint structure, not complete model accuracy.

6. Preview, launch workers before the head, and check actual inference:

   ```bash
   python3 scripts/cluster.py dry-run configs/cluster.local.json
   python3 scripts/cluster.py launch configs/cluster.local.json
   python3 scripts/cluster.py smoke configs/cluster.local.json
   python3 runtime/smoke-vision.py --base http://127.0.0.1:8000/v1 --out .local/vision-check-1
   ```

   The model alias is `deepseek-v41-flash`; default API base is `http://127.0.0.1:8000/v1`. Set `api_host` to a chosen network binding when using a separate benchmark client. The launcher waits up to 30 minutes for readiness and retains containers/logs on failure.

## Operation

```bash
python3 scripts/cluster.py status configs/cluster.local.json
python3 scripts/cluster.py stop configs/cluster.local.json
```

Configuration staging refuses to overwrite `node.env`. A new launch uses a new attempt number and deployment directory. Stop the previous deployment with its original configuration before launching a replacement. Containers are retained for diagnosis; removal is a separate Docker operation.

For the matched community prompt suite, run from a quiet client host:

```bash
bash bench/run-community.sh http://HEAD:8000/v1 results/my-unique-run spark8-matched
```

This executes the unchanged C1–C6 prompt benchmark and all four original cold-prefill targets, matching the boot 10 workload. Use a fresh server process for cold-cache comparisons: changing the output label does not change the deterministic prompt prefixes. Save the resolved server configuration with the results.

## Attribution and license

The existing vLLM project glue is MIT licensed; vLLM patch material retains Apache-2.0 terms. The added [SGLang subtree](sglang/NOTICE) contains AGPL-3.0-or-later Mia-derived adaptation, Apache-2.0 SGLang source and MIT benchmark material. Upstream benchmark and build credits are in [NOTICE](NOTICE), with license texts in [licenses](licenses/). Model weights and dependencies retain their own licenses.
