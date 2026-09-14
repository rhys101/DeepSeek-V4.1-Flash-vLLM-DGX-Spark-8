# DeepSeek V4.1 Flash on 8 × DGX Spark

Run DeepSeek V4.1 Flash across eight NVIDIA DGX Sparks with SGLang, native
resident Engram and an OpenAI-compatible API.

## Performance

Measured **14 September 2026** on the promoted **SG18 native prefill · TP8/EP4 · RoCEnante** build:

| Workload | 1 request | 8 concurrent requests |
|---|---:|---:|
| Coding | **131.72 tok/s** decode | **508.79 tok/s** total |
| Prose | **87.30 tok/s** decode | **254.65 tok/s** total decode |

Coding uses 200 output tokens; prose uses 256. Single-request decode excludes
first-token time. Coding aggregate includes prefill and batch time; prose uses
the output window. These are repeat-run means from the promoted build.

**Cold prefill** — input tokens per second:

| Input tokens per request | 1 request | 8 concurrent requests, aggregate |
|---|---:|---:|
| 4,096 | 4,428 tok/s | — |
| 32,768 | 4,196 tok/s | 4,236 tok/s |
| 131,072 | 3,907 tok/s | 3,944 tok/s |
| 299,008 | 3,458 tok/s | — |

Each rate divides input tokens by the mean of three cold trials, timed through
the last request's first token, after one excluded warmup. Reported cache hits
were zero. **64K has not been measured.** The three primary long-prompt cells
took **21.40–28.05% less time** in geometric mean against the controls before
and after the candidate.

The capacity repeat passed **eight distinct 997,100-token prompts active
together**: 8/8 exact cold retrievals, followed by 8/8 cached responses of 1,024
tokens each. The cold batch averaged **1,772 input tok/s** over 75.01 minutes;
the cached batch took 60.32 seconds. Minimum sampled OS-available memory was
**3.55 GiB** under the requested 1 GiB guard, after a Spark1 reboot.

[Method, all trials, earlier failed attempt and quality limits](sglang/docs/sg18-prefill-results.md)
· [Exact source and integration](sglang/experiments/sg18-prefill-tp-split/)
· [Development history](docs/progress.md).

## Eight Sparks and a switch

- **Compute:** 8 × DGX Spark, each with a GB10 GPU and 128 GB unified memory.
- **Network:** a switched Ethernet/RoCE fabric, with two fabric interfaces per
  node and MTU 9000. Configure the RDMA HCAs and interfaces for your hardware.
- **Layout:** Spark 1 hosts the API and rank 0; Sparks 2–8 host ranks 1–7.
  TP8 spans all nodes; EP4 groups the experts across pairs of ranks.
- **Storage:** the pinned DeepSeek checkpoint must be local on every Spark
  (about 510 GB per copy, plus image/build space). Engram stays resident during inference.

[Hardware and network details](docs/hardware-and-network.md).

## Quick start

The commands below use the **packaged SG5 launcher**. To reproduce the headline
SG18 native-prefill build, follow the [source and integration instructions](sglang/experiments/sg18-prefill-tp-split/);
its public installer is not yet automated.

Run on Spark 1 with Linux ARM64, Docker/NVIDIA GPU support, Buildx, Python 3.11+,
Git, SSH and rsync. Set up the RoCE fabric and model copies first. Build and
kernel checks require idle GPUs; stop any existing deployment before starting.

```bash
git clone https://github.com/rhys101/DeepSeek-V4.1-Flash-vLLM-DGX-Spark-8.git
cd DeepSeek-V4.1-Flash-vLLM-DGX-Spark-8/sglang
cp configs/cluster.example.json configs/cluster.local.json
```

Edit `cluster.local.json` with your eight hosts/IPs, SSH users, network
interfaces/HCAs, model path and a fresh deployment directory. The API defaults
to loopback; set `api_host` to a reachable address for remote clients.

Build, distribute, validate and launch:

```bash
bash scripts/build.sh configs/cluster.local.json
python3 scripts/cluster.py distribute --config configs/cluster.local.json
python3 scripts/cluster.py stage --config configs/cluster.local.json
python3 scripts/cluster.py preflight --config configs/cluster.local.json
python3 scripts/cluster.py dense --config configs/cluster.local.json
python3 scripts/cluster.py nccl --config configs/cluster.local.json
python3 scripts/check-kernels.py --config configs/cluster.local.json
python3 scripts/cluster.py serve --config configs/cluster.local.json
python3 validation/acceptance.py --base http://127.0.0.1:8000/v1 --out .local/acceptance-1
```

The launcher starts workers before the head and waits for readiness. The API
serves at `http://127.0.0.1:8000/v1` with model name `deepseek-v41-flash`.
If you changed the bind address, use that address for the acceptance check too.

```bash
python3 scripts/cluster.py status --config configs/cluster.local.json
python3 scripts/cluster.py stop --config configs/cluster.local.json
python3 scripts/cluster.py start --config configs/cluster.local.json
```

[Full deployment guide](docs/getting-started.md) · [SG18 native-prefill integration](sglang/experiments/sg18-prefill-tp-split/)
· [Earlier vLLM deployment](docs/vllm-deployment.md).

## Credits and licenses

Built on SGLang and [Mia’s Spark adaptation](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks),
with [Tony’s benchmark](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark).
See [NOTICE](NOTICE) and [SGLang licenses](sglang/NOTICE) for the AGPL, Apache and
MIT components. Model weights retain their upstream license.
