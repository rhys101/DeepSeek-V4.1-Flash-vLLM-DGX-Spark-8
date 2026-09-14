# SG18 indexer, mHC and WO source snapshot

This is the exact v06 combined kernel build, layered on the qualified SG17
TP8/EP4 RoCEnante deployment. [README qualification results](../../docs/sg18-indexer-mhc-wo-results.md)
and the [prior long-context and 4K study](../../docs/sg18-prior-study.md) use different workloads.

## Source and settings

The [sixteen production overlays](source/) preserve every file mounted by the
measured build. Eight files change or are added relative to SG17; the remaining
eight are inherited byte-for-byte. The [manifest](source-manifest.json) records
all production hashes, before/after hashes for the changes and the
[patch](sglang-indexer-mhc-wo.patch) hash. Paths in the patch start at the
SGLang `python/` directory; source snapshot paths start at the repository root.
Files retain their notices and Apache-2.0 terms; see [LICENSE.sglang](../../LICENSE.sglang).

Reuse SG17's exact [B12x source archive and manifest](../sg17-rocenante/), native
Engram runtime hooks, native query heads, base image and dependency composition.
All 241 B12x source files are unchanged. The [configuration renderer](configuration.py)
is also byte-identical to the measured SG17 renderer. Adapt the documentation
addresses and paths in [profile.example.json](profile.example.json).

For **every rank**, merge [environment.json](environment.json) into the result of
`configuration.environment(config, rank)` before creating the container. The
empty worker/partition strings select the qualified shape-dependent defaults;
retain `SPARK_MHC_WARPS=8`, `SPARK_WO_SPLITS=2` and `SPARK_WO_QUANT=1`.
Use the unchanged `engine_args(config, rank)` for TP8/EP4, 8M logical KV tokens,
128 request slots, a one-million-token context limit, 2,048-token prefill chunks
and static-five DSpark. Keep the two-GiB OS-available reserve guard on every node.

The runtime uses host networking/IPC, GPU and InfiniBand access, native
CUDA-resident Engram and the SG17 RoCEnante setup. Apply and hash-check all
source changes while the service is stopped, prepare the expert autotune cache
for the actual EP4 target/draft shapes, and verify source identity, native
Engram, EP4 geometry, captured expert dispatch and current transport/health
markers on all eight ranks before using performance numbers. Preserve the
previous deployment and its cache for rollback.

The standard Dockerfile and `scripts/cluster.py` package SG5. This source and
configuration snapshot describes the measured eight-node integration; a fresh
deployment through a generalized public SG18 installer has not been tested.

## Measurement

Use the existing, byte-identical [SG17 benchmark client](../sg17-rocenante/benchmark-client.py)
and pinned Tony coding / sparkDash prose implementations. The result report
records warmups, trial order, long-context checks and the fresh SG17 controls.
No benchmark arithmetic or prompt changes were made for SG18.

The WO-A GEMM changes reduction order. Component checks and capability smokes
do not establish broad semantic parity: prior combined-build code scores were
86/96 versus original 88/96 and 89/96, and exact answers were 51/96 versus 52/96
and 54/96. The prior study retains paired differences and original variability.

The public evidence can be checked without a GPU from the repository root:

```bash
python3 sglang/experiments/sg18-indexer-mhc-wo/verify-results.py
```

This verifies the 16 production files, all 35 unchanged benchmark reference files,
164 measured coding requests, 156 prose requests, published trial summaries,
C128 arithmetic and long-context answer receipts. Exact prose aggregate-window
reconstruction remains unavailable because the original runner omits its endpoints.
