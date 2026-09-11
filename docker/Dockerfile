# Reproduces the validated reference-base / feature-source / FlashInfer recipe.
ARG BASE_IMAGE=vllm/vllm-openai@sha256:a551e05307cd2e0092139d84db32af9c97e67d2eeeff072d21e429131d8c23f0
FROM ${BASE_IMAGE} AS stable-builder
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ARG VLLM_SHA=e47aa780bccf59f59dfa2cbb18e17a10b4fe69ba
RUN mkdir -p /src && curl -fSL --retry 3 --max-time 900 "https://codeload.github.com/vllm-project/vllm/tar.gz/${VLLM_SHA}" | tar xz -C /src --strip-components=1
COPY docker/build-stable-inner.sh /src/stable-inner.sh
RUN bash /src/stable-inner.sh

FROM ${BASE_IMAGE} AS runtime
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
COPY --from=stable-builder /src/vllm/ /usr/local/lib/python3.12/dist-packages/vllm/
COPY --from=stable-builder /src/build/_C_stable_libtorch.abi3.so /usr/local/lib/python3.12/dist-packages/vllm/_C_stable_libtorch.abi3.so
COPY patches/reference-attention.patch patches/manifest.json /opt/spark8-patches/
COPY scripts/apply-reference-patches.py /opt/apply-reference-patches.py
RUN python3 /opt/apply-reference-patches.py /opt/spark8-patches/manifest.json
COPY patches/small-m-mxfp8.patch patches/mxfp8-manifest.json /opt/spark8-patches/
RUN python3 /opt/apply-reference-patches.py /opt/spark8-patches/mxfp8-manifest.json
COPY versions.lock.json /opt/spark8-versions.lock.json
# Keep only the small upstream test subset used to validate the new extension.
RUN mkdir -p /opt/spark8-tests/tests/kernels && touch /opt/spark8-tests/tests/__init__.py /opt/spark8-tests/tests/kernels/__init__.py
COPY --from=stable-builder /src/tests/kernels/test_fused_deepseek_v4_qnorm_rope_kv_insert.py /opt/spark8-tests/tests/kernels/
COPY --from=stable-builder /src/tests/kernels/utils.py /opt/spark8-tests/tests/kernels/
COPY --from=stable-builder /src/tests/kernels/quant_utils.py /opt/spark8-tests/tests/kernels/
ARG FI_SHA=07869c61ba581e6d6b8ad8d142f4a6c89b707cc1
ARG CUTLASS_SHA=b46b16d003484063bca4ed365e44095c4c6ed633
ARG CCCL_SHA=16bd510c9b712e82b0ab6cbb630d8e29ba1f7116
ARG SPDLOG_SHA=c3aed4b68373955e1cc94307683d44dca1515d2b
RUN pip uninstall -y flashinfer-jit-cache flashinfer-cubin flashinfer-python \
 && mkdir -p /opt/fi-src \
 && curl -fSL --retry 3 --max-time 900 "https://codeload.github.com/flashinfer-ai/flashinfer/tar.gz/${FI_SHA}" | tar xz -C /opt/fi-src --strip-components=1
RUN mkdir -p /opt/fi-src/3rdparty/cutlass /opt/fi-src/3rdparty/cccl /opt/fi-src/3rdparty/spdlog \
 && curl -fSL --retry 3 --max-time 900 "https://codeload.github.com/NVIDIA/cutlass/tar.gz/${CUTLASS_SHA}" | tar xz -C /opt/fi-src/3rdparty/cutlass --strip-components=1 \
 && curl -fSL --retry 3 --max-time 900 "https://codeload.github.com/NVIDIA/cccl/tar.gz/${CCCL_SHA}" | tar xz -C /opt/fi-src/3rdparty/cccl --strip-components=1 \
 && curl -fSL --retry 3 --max-time 900 "https://codeload.github.com/gabime/spdlog/tar.gz/${SPDLOG_SHA}" | tar xz -C /opt/fi-src/3rdparty/spdlog --strip-components=1 \
 && test -f /opt/fi-src/3rdparty/cutlass/include/cutlass/cutlass.h \
 && test -f /opt/fi-src/3rdparty/cccl/README.md \
 && test -f /opt/fi-src/3rdparty/spdlog/include/spdlog/spdlog.h \
 && cd /opt/fi-src && BUILD_NVEP=0 FLASHINFER_BUILD_NO_PIP=1 pip install --no-deps --no-build-isolation .
ENV TORCH_CUDA_ARCH_LIST=12.1a FLASHINFER_CUDA_ARCH_LIST=12.1a \
    FLASHINFER_DISABLE_VERSION_CHECK=1 VLLM_HAS_FLASHINFER_CUBIN=1 \
    FLASHINFER_NVCC_THREADS=1 MAX_JOBS=2 \
    FLASHINFER_WORKSPACE_BASE=/opt/flashinfer-cache \
    VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_PLUGINS=""
RUN python3 -c "import flashinfer; from flashinfer.mla import supported_sparse_mla_sm120_configs as f; c=f()['dsv4']; assert flashinfer.__version__ == '0.7.0rc1'; assert c.supports_decode(num_heads=8, topk=1152); print('FlashInfer TP8 vision config OK', flashinfer.__version__)"
COPY docker/prewarm-fi07.py /opt/prewarm.py
RUN timeout 3000 python3 /opt/prewarm.py
LABEL org.opencontainers.image.title="DeepSeek V4.1 Flash on eight DGX Sparks" \
      ai.spark8.vllm-source="e47aa780bccf59f59dfa2cbb18e17a10b4fe69ba" \
      ai.spark8.flashinfer-source="07869c61ba581e6d6b8ad8d142f4a6c89b707cc1" \
      ai.spark8.reference-recipe="ca662ac35193c69ace9cee37f13a94abf2eff0fc"
