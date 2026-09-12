# SG17 RoCEnante source and configuration snapshot

This directory preserves the source and settings behind **134.91 coding decode tok/s at C1** and **474.52 coding aggregate tok/s at C8** on eight DGX Sparks. [Measured results and limitations](../../docs/sg17-rocenante-results.md).

The standard `scripts/cluster.py` and Dockerfile still package SG5. This is an experimental source snapshot and configuration renderer, not a separately validated one-command SG17 installer. The serving evidence comes from the original eight-node experimental deployment; a fresh deployment through generalized public wrappers has not been tested.

## Contents and composition

- [Three SGLang overlays](source/) and their [reviewable patch](sglang-rocenante.patch): global TP8 small SUM routing, graph capture and result-boundary health checks. They are Apache-2.0 SGLang-derived source, modified 12 September 2026.
- [Overlay manifest](overlay-manifest.json): original and final hashes for the three additions, plus the six unchanged SG11 overlays already present in [`../../patches/source/`](../../patches/source/). The two native-query-head files in that standard package also remain required.
- [Frozen B12x source archive](b12x-source.tar.gz), [241-file manifest](b12x-source-manifest.json), [composition record](b12x-local-patch.json) and [archive provenance](source-provenance.json). File bytes match the qualified bundle; only archive ownership/timestamps were normalized. [Apache-2.0 license](LICENSE.b12x) is also inside the archive.
- [Original pure configuration renderer](configuration.py) and [profile example](profile.example.json), with documentation-only addresses and generic paths. The example's archive hash describes the normalized public archive; the original observed archive hash is in the provenance receipt.
- [Benchmark wrapper](benchmark-client.py), with portable endpoint and repository paths. Request generation, warmups, trial order and metric computation follow the measured procedure.

The B12x archive retains the previously qualified `85d3681` kernels/compiler and local clamp fixes, with eight RoCEnante files from [081b235931dbbcedcf0eb5899bae990c5dec5238](https://github.com/local-inference-lab/b12x/tree/081b235931dbbcedcf0eb5899bae990c5dec5238) and two registry entries. It is a composed snapshot, not that complete upstream revision. Neither target nor draft uses the experimental B12x MoE backend.

## Integration requirements

Start from the pinned native-head image composition recorded in the report, retain the six shared production overlays and native Engram hooks, and verify original hashes before applying the three new source files. The public SG5 image has matching final shared source files, but an SG17 run built on that public image has not been separately measured.

The configuration renderer provides `engine_args(config, rank)` and `environment(config, rank)` for all eight ranks. Load the adapted profile through `load(path)` before rendering. Mount the qualified bundle at `/opt/sglang8/b12x`, retain the adapter/runtime import paths, and provide writable per-node cache directories. The recorded containers used host networking/IPC, GPU access, `/dev/infiniband`, `IPC_LOCK` capability and unlimited locked memory. The observed fabric used both `rocep1s0f0` and `roceP2p1s0f0`, port 1, RoCE v2 GID index 3; hardware-specific names must match the actual fabric. RoCEnante prepares BF16/FP32 kernels outside graph capture and enables routing only for the global eight-rank TP group.

Keep the measured TP8/EP4, 8M pool, 128 slots, 1M context cap, 2,048-token prefill chunks and five-token speculation settings. Enforce the 2 GiB OS-available reserve per node throughout startup and workload validation. Verify native Engram, target/draft FlashInfer EP4 geometry and `ROCE_TP8_READY`, `ROCE_TP8_ROUTE` and `ROCE_POST_RESULT_HEALTH` markers on all ranks. Follow the component, fault-handling, capability and long-context qualification in the report before accepting a fresh deployment.

## Repeat the measured client workload

On a quiet client with Python 3 and Node 22.19+, install the existing prose runner dependencies with `npm ci` from `sglang/bench/sparkdash`. From the repository root, after validating a compatible SG17 server:

```bash
python3 sglang/experiments/sg17-rocenante/benchmark-client.py \
  --base http://HEAD:8000/v1 --label sg17-local \
  --out /absolute/path/to/new-result-directory
```

The output directory must not exist. The wrapper verifies the unchanged community source hash and measured server settings, retains five C1 and three C4/C8 coding trials plus all excluded coding warmups, then runs three prose trials at each concurrency. It deliberately reuses prefixes without flushing the cache. Run the long-context and capability checks between initial and repeat suites to match the published sequence; see the report for the complete ordering.
