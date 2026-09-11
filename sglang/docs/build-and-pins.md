# EP4 source and build identity

The model-serving source is pinned to SGLang `e087e662ba1ac4ef7747537e2a9141085efd4561` in the Linux ARM64 image `lmsysorg/sglang@sha256:3475d88ec3124867d9d6f7b3bd49afdcf8ef4d6a9f8454fab331d5adb5204de7`. The observed stack declares FlashInfer 0.6.18, CUDA 13.0.3 and NCCL 2.30.7. [Version record](../versions.lock.json).

The original EP4 experiment used image ID `sha256:e7681b5276525821f9be286ed3bf0f51acb5239da27f69f60fdeb57e68c05581`, containing the native-head changes, with six production files and the runtime configuration mounted read-only. All eight ranks checked the mounted file hashes. An image ID by itself therefore does not identify that deployed composition.

The public [Dockerfile](../docker/Dockerfile) packages the same final eight SGLang source files directly into one image. [The manifest](../patches/manifest.json) records the original pinned-base hash and final hash for every file. The attention file's two historical steps—Mia's page split, then native heads—are collapsed into its exact final source. The installer checks **all** originals and replacements before copying any of them.

| Source group | Included changes |
|---|---|
| Two native-head files | `deepseek_v4.py` and `flash_mla_sm120.py`; H8/H16 dispatch with the V4.1 zero-filled fallback, reused buffers and page splitting retained |
| Five #39068 files | Ratio-2 verification compression, index postprocessing, candidate-block processing and backend wiring |
| DSpark worker | Existing speculative backend context applied to draft construction, graph capture and execution; actual all-rank methods/shapes recorded |
| Python hooks | Native Engram residency assertions, small-M dense b12x routing, low-ratio indexer metadata and long-prefill allocator workaround |
| Scheduler configuration | Minimum free slots set to one; the earlier experimental admission hook remains disabled |

No B12x **expert** backend is included. The dense hook's use of FlashInfer's b12x kernels is a separate implementation. The TP4 WO-A and Q-RoPE changes from the broader verification PR are excluded.

The model remains TP8. Four expert groups each span two tensor ranks (EP4/MoE-TP2). Each target rank owns 96 experts and each draft rank owns 32, with intermediate width 1,152. Packed weights/scales are divided without requantization. The model already performs one global TP8 post-expert reduction; no #32963 backport is included.

The packaged EP4 image built successfully on ARM64 as `sha256:fd439c24c2a11e7c43a750c55c104092fce691f652142181b0b16605a2a65c89`. Its eight actual final source hashes match the release manifest, and the draft-context checks pass. The seven unchanged attention/verification files match the SG3 package that passed SM121 tests immediately beforehand. [Package evidence](../results/ep4/package-validation/state.json).

The packaged host scripts are generalized from the experiment's launch/distribution scripts. They use an editable, gitignored configuration and dedicated `sglang8-ep4-r0` through `r7` container names. The full-model measurements use the preserved original EP4 deployment. A completely fresh eight-node installation through these public wrappers has not been separately reproduced; successful build/kernel checks do not imply that it has.

The model checkpoint revision is `df42c109f1defefcbfcedbe7d905718a12266e40`; the launcher verifies its `config.json` hash. This is not a full checksum verification of all model tensors. Obtain the complete checkpoint on every node using your normal model-download workflow. Build sources are pinned, but this is not a hermetic package lock or a public prebuilt container distribution.

Mia's repository uses a mutable image tag, so the exact software image behind her reported figures is unknown. Our pinned image is the ARM64 resolution used for this work, not evidence of her exact benchmark image.
