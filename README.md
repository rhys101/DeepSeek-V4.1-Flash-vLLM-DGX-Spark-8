# DeepSeek V4.1 Flash on eight DGX Sparks — SGLang EP4

**495.32 coding tokens/s across eight concurrent requests · 131.05 coding decode tokens/s on one request.**

Measured **14 September 2026** on eight DGX Sparks with **SGLang SG18, TP8/EP4 and RoCEnante**, adding bounded indexer, mHC and WO kernels. The repeated README suite is **4.38% above the published SG17 C8 throughput**. Both C8 runs exceed the fresh SG17 controls and published C8 result. **C1 decode is 2.86% lower than SG17’s 134.91 tok/s record**; SG17 remains the faster single-request reference. C1 is per-stream decode; C8 is full-batch aggregate including prefill. [All four runs, every trial and method](sglang/docs/sg18-indexer-mhc-wo-results.md).

SG18 retains native resident Engram, **8M logical KV tokens, 128 request slots, a 1M-token context limit and five-token speculation**. It passed **128/128 concurrent arithmetic requests**, image/JSON/tool checks and exact retrieval through **299,098 input tokens**. Earlier bounded long-context quality scores remain lower on some tasks; broad quality parity is not established. [Source and integration settings](sglang/experiments/sg18-indexer-mhc-wo/) · [Quality results and limits](sglang/docs/sg18-prior-study.md).

The earlier SG11 concurrency sweep reached **1,250.30 coding tok/s** and **774.68 prose tok/s at C128**; those remain separate historical measurements. SG18 throughput has been measured at C1/C4/C8. [SG11 C8–C128 results](sglang/docs/concurrency-results.md).

The standard build and launcher below package **SGLang EP4 (SG5)**: **TP8/EP4, native RAM-resident Engram, DSpark five-token drafting, CUDA graphs, eight request slots, four images and a 300,000-token context cap**. It runs the same checkpoint as the [vLLM deployment](docs/vllm-deployment.md), whose complete build guide and earlier measurements remain available. The SGLang source, scripts and evidence are in [`sglang/`](sglang/).

It combines [Mia's pinned Spark adaptation](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks/tree/e59e6eb67479aa68f6fa700c600dc90a0729b5ec) with native-width query heads from [SGLang #36655](https://github.com/sgl-project/sglang/pull/36655), the scheduler's `--min-free-slots-delay 1` setting, and five verification/index-processing files from [#39068](https://github.com/sgl-project/sglang/pull/39068). Each of four expert groups spans two tensor ranks; model-wide TP remains eight. A local draft-context fix applies the requested backend consistently and records the actual loaded expert layout on all ranks. [SG3](sglang/docs/sg3-reference.md) is retained as the earlier reference.

## Speed

### SG18: indexer, mHC and WO kernels

| Concurrency | Coding decode per stream (tok/s) | Coding full-batch aggregate (tok/s) | Prose aggregate decode (tok/s) |
|---|---|---|---|
| C1 | 131.05 | 117.49 | 86.89 |
| C4 | 83.22 | 299.71 | 166.76 |
| C8 | 68.33 | 495.32 | 253.84 |

These are means from the repeat suite after long-context and capability checks: five C1 coding trials, three C4/C8 coding trials, and three prose trials per concurrency. Initial SG18 coding means were **131.28 tok/s at C1** and **505.25 tok/s at C8**. All initial/repeat SG17 and SG18 results, including prose and slower trials, are retained. [Complete comparison](sglang/docs/sg18-indexer-mhc-wo-results.md).

The gains depend on the workload: C1/C8 prose improved, while C4 prose fell from **180.46** to **166.76 tok/s**. C1 coding and C4 coding aggregate were also lower than the published SG17 repeat.

### Earlier SG17: single-request decode with RoCEnante

| Concurrency | Coding decode per stream (tok/s) | Coding full-batch aggregate (tok/s) | Prose aggregate decode (tok/s) |
|---|---|---|---|
| C1 | 134.91 | 121.86 | 80.99 |
| C4 | 83.67 | 305.00 | 180.46 |
| C8 | 64.89 | 474.52 | 244.15 |

These are means from the second complete SG17 benchmark on the same server process, after the long-context and capability checks: five coding trials at C1, three at C4/C8, and three prose trials per concurrency. Coding decode excludes first-token latency; coding aggregate includes the full batch; prose uses its own first-to-last-output timing. The initial C8 coding aggregate mean was **470.58 tok/s**. [Both complete runs, latency, validation and limitations](sglang/docs/sg17-rocenante-results.md).

### Earlier SG11 short-prompt concurrency with an 8M KV pool

Measured **12 September 2026**. All rows below were measured on the same **SG11 TP8/EP4 experimental profile: 128 request slots, a 1,000,000-token context limit and 2,048-token prefill chunks**. The configured KV pool remained **8,000,000 tokens** throughout, with a **2 GiB OS-available reserve floor on every Spark**. These settings are separate from the packaged SG5 launcher defaults below.

| Concurrency | Coding aggregate tok/s | Coding per-stream decode tok/s | Prose aggregate decode tok/s | Prose per-stream decode tok/s |
|---|---|---|---|---|
| C8 | 445.46 | 60.98 | 231.59 | 31.13 |
| C16 | 689.38 | 48.81 | 376.08 | 25.16 |
| C32 | 975.92 | 35.07 | 553.14 | 18.58 |
| C64 | 1216.86 | 21.89 | 698.20 | 11.84 |
| C128 | 1250.30 | 11.15 | 774.68 | 6.57 |

Moving from C64 to C128 changed coding aggregate throughput by **+2.7%** and prose by **+11.0%**, with lower per-stream speeds.

At C128, average first-token latency was **1.906 s for coding** and **1.543 s for prose**. Concurrent inputs were 30–303 tokens; separate prefill probes stayed below 128K. All eight containers remained running with **no OOMs or restarts**. Lowest observed OS-available memory on an individual Spark was **6.89 GiB during startup** and **8.71 GiB during benchmarking**, with **no additional OS swap use during benchmarking**.

The SG11 profile passed text, one-image and four-image, structured JSON and tool-call capability checks, plus **128/128 exact arithmetic requests at C128**. A subsequent audit confirmed that the C64/C128 summaries match the raw benchmark evidence, recorded configuration, source hashes, memory samples and observed concurrency. These are capability smokes, not a broad model-quality evaluation; the short-prompt run does not validate full 8M-pool capacity or 128 simultaneous million-token contexts.

Coding aggregate includes prefill and full batch time; prose aggregate uses the first-to-last-output window. Prose is the mean of two trials, using the documented one-line C64/C128 allowlist extension. The short requests did not fill the 8M pool. [Full results, C64 run, latency, configuration and evidence](sglang/docs/concurrency-results.md).

### Standard-profile engine comparison

**Why the C8 figures differ:** C8 means eight active requests, not the server’s maximum slot count. The table above measures SG11 (128 slots, 8M KV pool, 1M context limit, 2,048-token prefill chunks); this earlier engine comparison measures SG5 (eight slots, 3.2M KV pool, 300K context limit, 8,192-token prefill chunks). Coding C8 is **445.46 tok/s on SG11 versus 431.82 tok/s on SG5**. Both use the community coding workload and full-batch aggregate timing, but they are separate measurements under different configurations. The 3.2% difference does not establish a tuning gain: coding has no repeated-run confidence intervals. The corresponding eight-category C8 means are **290.63 and 291.33 tok/s**, respectively. Retain the SG5 result for the recorded vLLM comparison and use SG11 for C8–C128 scaling.

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

The engine comparison uses the repository's existing, unmodified [community benchmark](bench/v41bench.py): the same coding prompt, deterministic request prefixes, 200-token coding budget, temperature zero and thinking off. Single-stream decode excludes time before the first token. Aggregate throughput includes prefill and batch wall time. These two metrics are reported separately.

### Cold prefill

The same one-token-reply workload, with identical prompt token counts. These times include the complete one-token response; rates divide input tokens by that time.

| Prompt tokens | vLLM time | SGLang EP4 time | vLLM input tok/s | EP4 input tok/s |
|---|---|---|---|---|
| 2,950 | 1.059 s | 1.187 s | 2,785.5 | 2,485.0 |
| 11,592 | 4.008 s | 3.715 s | 2,892.5 | 3,120.1 |
| 46,810 | 15.844 s | 12.606 s | 2,954.3 | 3,713.4 |
| 93,335 | 33.505 s | 21.905 s | 2,785.7 | 4,261.0 |

EP4 was slower on the smallest prefill and faster on the three larger cases. These are single measurements. [Raw results and method](sglang/docs/community-comparison.md#cold-prefill).

### Separate prose benchmark

The separate sparkDash **prose** benchmark reached **67.300 tok/s at C1, 154.395 at C4 and 228.640 at C8**, averaged across two trials. sparkDash uses a different prompt, a 256-token budget and aggregate decode timing. Its results must not be used as a direct comparison with the community coding figures. [Prose and prefill results](sglang/docs/prose-results.md).

### Eight-million-token capacity test

A separate TP8/EP4 profile passed **eight concurrent 997,097-token retrieval requests (7,976,776 input tokens total)**. Aggregate prefill peaked at **5,076 tok/s within the first prompt's initial 128K** and averaged **1,513 tok/s across the cold run**. Lowest sampled OS-available memory was **7.83 GiB on an individual Spark**. [Full results, cached output speed and C1/C8 append latency](sglang/docs/eight-million-token-results.md).

## Quality

The packaged SG5 profile passed text arithmetic, two waves of eight concurrent arithmetic requests, one-image and four-image checks, structured JSON, and a tool-call round trip. Exact three-record retrieval passed at **32,867, 131,171 and 299,099 actual prompt tokens**. These are capability smokes; broad model-quality parity with vLLM or an unmodified reference remains unmeasured.

Checkpoint MXFP4 expert weights, FP8 dense weights, the BF16 activation dtype, automatic KV selection (resolved to FP8 E4M3), BF16 WO-A computation and speculation acceptance thresholds are retained. SM121 expert computation uses the existing FlashInfer MXFP4/MXFP8 path; the name of the global activation dtype does not mean every GEMM computes in BF16. No additional quantization or TP4 WO-A/Q-RoPE changes are included.

The unchanged five-file verification/index composition passed **36 tests and four subtests** on GB10/SM121; four all-padded cases were skipped by the upstream suite. Native/padded H8/H16 numerical comparisons and H8 changed-input graph replay also passed. The EP4 partition probe passed 90 cases using real target/draft weights, lossless weight/scale partitioning, changed-input graph replay and repeated-output checks. Splitting experts changes floating-point partial-sum order; broad quality parity and bitwise parity are not established. [Validation evidence and limits](sglang/docs/validation.md).

During the original EP4 validation, all eight ranks stayed above **17.57 GiB OS-available memory**, without OOMs or restarts. That memory figure belongs to the original profile; the [separate capacity experiment](sglang/docs/eight-million-token-results.md) measured eight simultaneous near-million-token contexts.

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

## Reproduce the standard SG5 deployment

The headline SG18 result uses the separately preserved [SG18 source and integration settings](sglang/experiments/sg18-indexer-mhc-wo/). The commands below build and launch SG5; changing only its example JSON does not reproduce SG18.

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

   The build starts from a pinned ARM64 image and verifies all eight original/final SGLang file hashes before installing them. It writes **your own build's image ID** into the local configuration. Distribution verifies archive checksums and imported image IDs. [Source/build identity](sglang/docs/build-and-pins.md).

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

Record input/output token counts and the resolved server configuration with the result. Preserve the slower runs. The separate [sparkDash runner](sglang/bench/sparkdash/) requires Node 22.19+ and `npm ci`; its pinned upstream files are hash-checked by the CLI.

```bash
python3 scripts/cluster.py status --config configs/cluster.local.json
python3 scripts/cluster.py stop --config configs/cluster.local.json
python3 scripts/cluster.py start --config configs/cluster.local.json
```

`start` restarts the retained containers. Use a new deployment directory and remove old stopped containers explicitly when creating a new configuration with the same fixed EP4 container names. Do not operate the vLLM and SGLang launchers concurrently on these GPUs.

## Attribution and license

The `sglang/` subtree contains a Mia-derived **AGPL-3.0-or-later** adaptation, Apache-2.0 SGLang source and MIT benchmark material. See [NOTICE](sglang/NOTICE) and the included license texts. The existing vLLM material retains its own terms; see the [root notice](NOTICE). Model weights and dependencies keep their upstream licenses.
