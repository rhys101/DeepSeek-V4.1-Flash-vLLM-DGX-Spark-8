# Deployment guide

Run the cluster from Spark 1 (rank zero). The [main README](../README.md) provides
the short version; this page records the packaged configuration, validation and
operation commands.

The **SG18** headline deployment has its own [source and integration instructions](../sglang/experiments/sg18-indexer-mhc-wo/).
The standard Dockerfile and launcher below package **SG5**. A generalized public
SG18 installer has not been tested; editing the SG5 configuration alone does not
install SG18. [Benchmark and development history](progress.md).

## Standard SG5 configuration

| Setting | EP4 |
|---|---|
| Hardware | Eight GB10/SM121 DGX Sparks; one process/GPU per node |
| Parallelism | TP8, EP4, MoE-TP2, PP1; existing RoCE fabric |
| Model | `deepseek-ai/DeepSeek-V4.1-Flash`, revision `df42c109f1defefcbfcedbe7d905718a12266e40` |
| Engram | Native CUDA-resident owned rows, about 23.604 GiB raw weights/scales per Spark |
| KV cache | `auto`, resolved to FP8 E4M3 by the pinned runtime |
| Context / requests / images | 300,000 tokens / eight requests / four images |
| Prefill | 8,192-token chunks and max-prefill setting |
| Memory | Static fraction 0.80, fixed 3.2M-token pool, 13 GiB minimum OS reserve |
| Speculation | DSpark block size five; static verification; thresholds 1.0 |
| Expert / dense backends | FlashInfer MXFP4 experts; Mia's small-M b12x dense routing with CUTLASS fallback |
| Scheduler | `--min-free-slots-delay 1`; earlier admission-retry hook disabled |
| Decode graphs | Explicit request batch sizes 1–8 |
| NCCL | 1 MiB buffer, 256 KiB LL128 buffer, `^LL128`, eight channels |
| API alias | `deepseek-v41-flash` |

SGLang's pool and prefill accounting differ from vLLM's. Equal numeric flags do not prove identical memory use or scheduler behavior. Engram lookup arithmetic remains upstream; the hooks assert ownership/residency and retain the validated SM120 metadata/page-splitting and long-prefill allocation workarounds.

## Build, validate and launch

The headline SG18 result uses the separately preserved [SG18 source and integration settings](../sglang/experiments/sg18-indexer-mhc-wo/). The commands below build and launch SG5; changing only its example JSON does not reproduce SG18.

Run cluster operations on rank zero. Requirements: Linux ARM64 Sparks, Docker with NVIDIA GPU support and Buildx, Python 3.11+, SSH/rsync, a working RDMA fabric, and the pinned checkpoint already present on every node. Weights and credentials are not included. Build and kernel tests require idle GPUs; stop the active deployment with its own configuration first.

From the repository root, enter the SGLang directory. Run the setup and operation commands below from there:

```bash
cd sglang
```

1. Configure the eight nodes:

   ```bash
   cp configs/cluster.example.json configs/cluster.local.json
   ```

   Replace the documentation-only IP addresses, login users, fabric interface/HCA names, model directory and deployment directory. The example binds the API to loopback; select a reachable bind address for a separate benchmark client. Keep the EP4 model/performance settings unchanged for reproduction. The local configuration is gitignored.

2. Build and distribute one verified image:

   ```bash
   bash scripts/build.sh configs/cluster.local.json
   python3 scripts/cluster.py distribute --config configs/cluster.local.json
   ```

   The build starts from a pinned ARM64 image and verifies all eight original/final SGLang file hashes before installing them. It writes **your own build's image ID** into the local configuration. Distribution verifies archive checksums and imported image IDs. [Source/build identity](../sglang/docs/build-and-pins.md).

3. Stage a fresh deployment and check it:

   ```bash
   python3 scripts/cluster.py stage --config configs/cluster.local.json
   python3 scripts/cluster.py preflight --config configs/cluster.local.json
   python3 scripts/cluster.py dense --config configs/cluster.local.json
   python3 scripts/cluster.py nccl --config configs/cluster.local.json
   python3 scripts/check-kernels.py --config configs/cluster.local.json
   python3 scripts/cluster.py dry-run --config configs/cluster.local.json
   ```

   Staging refuses to overwrite an existing `kit` directory. Choose a new deployment directory for a new configuration. The dense check, collective check, and numerical kernel tests have separate purposes; a passing source check alone is not inference validation.

4. Launch workers before the head, wait for readiness, then test the API:

   ```bash
   python3 scripts/cluster.py serve --config configs/cluster.local.json
   python3 validation/acceptance.py --base http://127.0.0.1:8000/v1 --out .local/acceptance-1
   python3 validation/long-context.py --base http://127.0.0.1:8000/v1 --tag ep4-local-1 --out .local/long-context-1
   ```

   The launcher checks every rank and the OS reserve while loading. It stops the containers it started if startup fails or the reserve is breached, retaining the containers and logs. Continue monitoring available memory under your own workload.

## Benchmark and operate

Run the matched community suite from a quiet client. Start with a fresh server process/cache because the reference script's deterministic request tags repeat across runs. Choose a new output directory:

```bash
bash bench/run-community.sh http://HEAD:8000/v1 .local/community-1 ep4-matched
```

Record input/output token counts and the resolved server configuration with the result. Preserve the slower runs. The separate [sparkDash runner](../sglang/bench/sparkdash/) requires Node 22.19+ and `npm ci`; its pinned upstream files are hash-checked by the CLI.

```bash
python3 scripts/cluster.py status --config configs/cluster.local.json
python3 scripts/cluster.py stop --config configs/cluster.local.json
python3 scripts/cluster.py start --config configs/cluster.local.json
```

`start` restarts the retained containers. Use a new deployment directory and remove old stopped containers explicitly when creating a new configuration with the same fixed EP4 container names. Do not operate the vLLM and SGLang launchers concurrently on these GPUs.
