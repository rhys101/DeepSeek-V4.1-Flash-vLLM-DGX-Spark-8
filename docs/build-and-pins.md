# Build and source pins

The current recipe follows [Tony’s pinned reference recipe](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark/tree/ca662ac35193c69ace9cee37f13a94abf2eff0fc) for the ARM64 nightly base, feature-source overlay, rebuilt stable CUDA extension, FlashInfer, and four attention fixes. The current profile adds an original small-M MXFP8 routing patch informed by Mia's work. Native Engram remains resident in memory.

`scripts/build.sh` runs `docker/Dockerfile` on an idle Linux ARM64 Spark. Its first stage builds `_C_stable_libtorch.abi3.so` from the pinned feature source, with CMake 3.31.6 and eight compiler jobs. The runtime stage overlays the feature Python tree and that extension, verifies and applies both the reference attention patch and the small-M MXFP8 patch with zero fuzz, installs FlashInfer without dependency replacement, and compiles its SM12x MXFP8 and sparse MLA modules. GPU loading and execution are validated after building.

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

The running image was assembled from these stages, including a fresh stable extension built on a second idle Spark, then extended with the checked MXFP8 source layer. All eight nodes run the identical final image and source-file hash. The selected NCCL environment is applied by the runtime profile, not baked into the image. The packaged combined wrapper has not had a separate complete cold rebuild. Image identity will differ on rebuild; record your own ID in the cluster config.

## Small-M MXFP8 layer

`patches/mxfp8-manifest.json` checks the original and final SHA-256 of vLLM's `model_executor/kernels/linear/mxfp8/flashinfer.py`. The patch uses FlashInfer's existing `backend="b12x"` on SM12x when M is at most the configured threshold (128), K is divisible by 128, and output is BF16 or FP16. Other shapes retain CUTLASS. `SPARK_MXFP8_B12X_MAX_M=0` disables this routing on the next launch; the default profile enables 128.

The patch uses the existing swizzled MXFP8 scales and quantization; it does not change the checkpoint format, MoE backend or Engram placement. B12x requires the pinned CUDA 13 / CUTLASS DSL stack. Both routes and the M=128/129 boundary passed GPU integration and changed-input graph replay checks.

Use a new deployment/cache directory after a source change, as in the quick start. The measured upgrade used a fresh vLLM compilation-cache namespace. First boot also tunes the actual b12x shapes; wait for model and graph readiness before benchmarking. A full cold rebuild of the combined wrapper remains untested; record the image ID produced by your own build instead of expecting Docker metadata to reproduce the measured ID.

The standalone Torch collective probes reported NCCL 2.29.7. The serving process explicitly logs its dynamically loaded NCCL 2.30.7+cuda13.3. The probe and serving results are recorded separately.
