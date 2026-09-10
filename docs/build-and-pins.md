# Build and source pins

The current recipe follows [Tony’s pinned reference recipe](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark/tree/ca662ac35193c69ace9cee37f13a94abf2eff0fc) for the ARM64 nightly base, feature-source overlay, rebuilt stable CUDA extension, FlashInfer, and four attention fixes. Native Engram remains resident in memory.

`scripts/build.sh` runs `docker/Dockerfile` on an idle Linux ARM64 Spark. Its first stage builds `_C_stable_libtorch.abi3.so` from the pinned feature source, with CMake 3.31.6 and eight compiler jobs. The runtime stage overlays the feature Python tree and that extension, verifies and applies the reference attention patch with zero fuzz, installs FlashInfer without dependency replacement, and compiles its SM12x MXFP8 and sparse MLA modules. GPU loading and execution are validated after building.

| Component | Pin or observed version |
|---|---|
| ARM64 base | `vllm/vllm-openai@sha256:a551e05307cd2e0092139d84db32af9c97e67d2eeeff072d21e429131d8c23f0` |
| Base tag / distribution version | `nightly-8a728663c1c3eeace834a95f5654fa653cc1998c` / `0.28.1rc1.dev388+g8a728663c` |
| Actual vLLM feature source | `e47aa780bccf59f59dfa2cbb18e17a10b4fe69ba` (`dsv41-feat` as inspected) |
| FlashInfer | `0.7.0rc1`, source `07869c61ba581e6d6b8ad8d142f4a6c89b707cc1` |
| Torch / Transformers | `2.13.0+cu130` / `5.16.1` |
| CUTLASS DSL / TileLang / Triton | `4.6.2` / `0.1.12` / `3.7.1` |
| CUDA base environment / nvcc | `13.0.2` / release 13.0, `13.0.88` |
| NCCL / NVSHMEM | `2.30.7` / `3.4.5` |

**The printed vLLM distribution version identifies the base wheel, not the overlaid feature-source revision.** Tony’s exact checkout when his image was built is not recorded by that version string.

All source/submodule pins are in [versions.lock.json](../versions.lock.json). CMake requests architecture 12.1a; upstream CMake selects the compatible 12.0f family target for applicable stable kernels. The result was tested on SM121. Runtime JIT work is capped at two jobs and one FlashInfer NVCC thread; preserve `FLASHINFER_WORKSPACE_BASE=/opt/flashinfer-cache` to use the baked modules.

The official model snapshot remains `df42c109f1defefcbfcedbe7d905718a12266e40`. A comparison against current revision `dba1be0a40aa45a94ad051997016db3960a90277` found all 48 weight objects, configuration, tokenizer and index unchanged. Only three reference encoding files changed. Tony’s download script does not pin a snapshot, so his exact downloaded revision is unknown.

The running image was assembled from these stages, including a fresh stable extension built on a second idle Spark. The packaged combined wrapper has not had a separate complete cold rebuild. Image identity will differ on rebuild; record your own ID in the cluster config.
