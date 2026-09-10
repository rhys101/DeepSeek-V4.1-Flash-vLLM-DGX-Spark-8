# DeepSeek V4.1 Flash on eight DGX Sparks

Serve `deepseek-ai/DeepSeek-V4.1-Flash` across **eight NVIDIA DGX Sparks** with **TP8, RAM-resident Engram, DSpark k=5, CUDA graphs and vision**.

This repository contains the build recipe, serving configuration, validation tools and measured results for the deployment recorded on **10 September 2026**. The software follows [Tony’s pinned reference recipe](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark/tree/ca662ac35193c69ace9cee37f13a94abf2eff0fc), using FlashInfer 0.7.0rc1.

## Performance

**92.7 tok/s single-stream coding decode. Six concurrent requests is the practical throughput/latency recommendation:** 255.61 tok/s coding aggregate and 170.81 tok/s averaged across the eight benchmark categories. Eight concurrent requests reaches 187.80 tok/s category mean, with slower individual streams.

| Measurement (tok/s) | Tony, 4 Sparks | This deployment, 8 Sparks | Ratio |
|---|---:|---:|---:|
| Coding C1, decode per stream | 73.78 | 92.67 | 1.26× |
| Coding C4, aggregate | 123.80 | 241.99 | 1.95× |
| Coding C6, aggregate | 225.48 | 255.61 | 1.13× |
| Category mean C4, aggregate | 85.72 | 143.75 | 1.68× |
| Category mean C6, aggregate | 131.86 | 170.81 | 1.30× |

The category mean covers code, JSON, math, reasoning, tables, summary, prose and narrative; counting is excluded. C means concurrent requests. Aggregate throughput includes prefill, while per-stream decode excludes time before the first token.

At C8, mean aggregate throughput is 9.9% above C6 and mean per-stream decode is 18.5% lower (27.53 versus 33.78 tok/s). C6 is a practical tradeoff from these measurements, not a proven optimum for every workload. The server has eight request slots. Tony’s pinned boot 10 reference stops at C6.

### Cold prefill

Cold, unique prompts with a one-token reply. Actual input sizes match Tony’s four-Spark reference; rates are prompt tokens divided by time to first token (TTFT).

| Actual prompt tokens | Tony, 4 Sparks TTFT | This deployment TTFT | Tony prefill tok/s | This deployment prefill tok/s |
|---:|---:|---:|---:|---:|
| 2,950 | 3.270 s | 1.156 s | 902.2 | 2,551.5 |
| 11,592 | 11.293 s | 5.727 s | 1,026.5 | 2,024.0 |
| 46,810 | 30.426 s | 17.875 s | 1,538.5 | 2,618.7 |
| 93,335 | 78.173 s | 36.381 s | 1,194.0 | 2,565.5 |

The main run completed 54 category batches / 189 requests and these four prefill cases. C8 was measured separately with nine batches / 72 requests on the same configuration. All 189 category input counts match the reference; 82 output lengths differ. These are single measurements without repeated-run confidence intervals. A 300K context cap is configured; the longest measured prompt was 93,335 tokens, with no full 300K or long-context quality evaluation.

See [the comparison and concurrency tradeoff](docs/comparison.md), [complete benchmark tables](docs/benchmark-tables.md), [method and reproduction commands](docs/benchmark-method.md), and [raw results](results/).

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
| API model alias | `deepseek-v41-flash` |

The [profile](configs/profiles/dspark-300k.json) records the exact flags and graph sizes. The comparison uses twice Tony’s hardware: C6 category throughput is 1.30× his result, giving lower throughput per Spark. TP8/TP4, Engram placement and fabric selection differ; this does not isolate RAM-versus-disk performance.

All eight containers passed the recorded health checks with no OOM kills or restarts. Arithmetic, image OCR/colors/shapes and four-image ordering passed. [GPU validation](docs/validation.md) covers native kernels, sparse attention, prefill and graph replay; these checks are not a broad model-quality evaluation.

The measured memory footprint is about 73.16 GiB loaded model allocation (including 23.60 GiB Engram) and 15.29 GiB KV per Spark, within a GPU-worker counter of about 97.75 GiB. Linux reported 3.41–5.01 GiB available RAM. See [memory accounting](docs/memory-footprint.md); a utilization setting of 0.80 does not guarantee 20% available unified RAM.

## Requirements

Linux ARM64 DGX Sparks; Docker with NVIDIA GPU support and Buildx; Python 3.10+, Git, SSH and rsync; a working RDMA fabric with MTU 9000. The checkpoint must be available locally on every node: approximately 510 GB of tensor data per copy, plus space for build caches and images. See [hardware and networking](docs/hardware-and-network.md).

Source pins and observed package versions are in [versions.lock.json](versions.lock.json). They are not a hermetic package lock. The running image was assembled from validated build stages; the packaged combined wrapper and clean-cluster installation have not had a separate complete reproduction. No public prebuilt image is provided.

## Quick start

Perform cluster operations on rank 0. All inter-node SSH and rsync operations bind to its configured fabric address. Confirm SSH host keys and routes before beginning.

1. Build on an idle Spark:

   ```bash
   bash scripts/build.sh
   docker image inspect --format '{{.Id}}' deepseek-v41-spark8:2026-09-10-fi07-vision
   ```

   The build uses the pinned ARM64 nightly base, rebuilds the stable CUDA extension, overlays pinned feature source, applies the reference attention patch with zero fuzz and hash checks, installs FlashInfer 0.7, and compiles its two SM12x modules. Details: [build and pins](docs/build-and-pins.md).

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

This executes the unchanged C1–C6 prompt benchmark and all four original cold-prefill targets, matching the boot 10 workload. Save the resolved server configuration with the results.

## Attribution and license

Project glue is MIT licensed; vLLM patch material retains Apache-2.0 terms. Upstream benchmark and build credits are in [NOTICE](NOTICE), with license texts in [licenses](licenses/). Model weights and dependencies retain their own licenses.
