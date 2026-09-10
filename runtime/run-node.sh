#!/usr/bin/env bash
# Eight-Spark deployment runner. Individual kernels were checked before launch.
set -euo pipefail

action=${1:-help}
if [[ "$action" == help || "$action" == --help ]]; then
  echo 'Usage: bash run-node.sh preflight|nccl|serve|status|stop [node.env]'
  echo 'Read the deployment plan and edit node.env before using a Spark.'
  exit 0
fi
case "$action" in preflight|nccl|serve|dry-run|validate-config|status|stop) ;; *) echo 'Unknown action' >&2; exit 2 ;; esac
config=${2:-node.env}
if [[ ! -f "$config" ]]; then echo "Missing configuration: $config" >&2; exit 2; fi
source "$config"
: "${NODE_RANK:?}" "${IMAGE:?}"
[[ "$NODE_RANK" =~ ^[0-7]$ ]] || { echo 'NODE_RANK must be 0..7' >&2; exit 2; }
name="${CONTAINER_PREFIX:-deepseek-v41}-a${ATTEMPT:-1}-r${NODE_RANK}"
if [[ "$action" == stop ]]; then exec docker stop "$name"; fi
if [[ "$action" == status ]]; then
  docker inspect --format '{{.State.Status}} image={{.Image}} exit={{.State.ExitCode}}' "$name"
  docker logs --tail 60 "$name"
  exit 0
fi

for key in HEAD_IP NODE_IP FABRIC_IF NCCL_HCAS MODEL_STORE MODEL_SUBPATH RUN_DIR; do
  value=${!key:-}
  if [[ -z "$value" || "$value" == *CHANGE_ME* || "$value" == /absolute/path/* ]]; then
    echo "Set $key in $config first" >&2; exit 2
  fi
done
[[ "$MODEL_STORE" == /* && "$RUN_DIR" == /* ]] || { echo 'Use absolute host paths' >&2; exit 2; }
[[ "$MODEL_SUBPATH" != /* && "/$MODEL_SUBPATH/" != *'/../'* ]] || { echo 'MODEL_SUBPATH must stay inside MODEL_STORE' >&2; exit 2; }
[[ -d "$MODEL_STORE/$MODEL_SUBPATH" ]] || { echo 'Checkpoint directory is missing' >&2; exit 2; }
[[ "$(uname -m)" == aarch64 ]] || { echo 'Run on a Linux ARM64 Spark' >&2; exit 2; }
[[ -d /dev/infiniband ]] || { echo 'RDMA devices are not exposed on this host' >&2; exit 2; }
ip link show dev "$FABRIC_IF" >/dev/null
docker image inspect "$IMAGE" >/dev/null || { echo "Pull the pinned image first: docker pull $IMAGE" >&2; exit 2; }
if [[ -n ${EXPECTED_IMAGE_ID:-} ]]; then
  [[ $(docker image inspect --format '{{.Id}}' "$IMAGE") == "$EXPECTED_IMAGE_ID" ]] || { echo 'Image ID does not match this deployment' >&2; exit 2; }
fi

kit_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "$RUN_DIR/cache"
model_path="/models/$MODEL_SUBPATH"
common=(
  --pull never --gpus all --network host --ipc host
  --device /dev/infiniband --cap-add IPC_LOCK
  --ulimit memlock=-1 --ulimit nofile=1048576:1048576
  --mount "type=bind,src=$MODEL_STORE,dst=/models,readonly"
  --mount "type=bind,src=$RUN_DIR/cache,dst=/cache"
  --mount "type=bind,src=$kit_dir,dst=/kit,readonly"
  -e "VLLM_HOST_IP=$NODE_IP"
  -e "NCCL_SOCKET_IFNAME==$FABRIC_IF"
  -e "GLOO_SOCKET_IFNAME=$FABRIC_IF"
  -e "NCCL_IB_HCA=$NCCL_HCAS"
  -e NCCL_IB_DISABLE=0 -e NCCL_DEBUG=INFO
  -e NCCL_DEBUG_SUBSYS=INIT,NET
  -e CUDA_VISIBLE_DEVICES=0
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600
  -e VLLM_ALLREDUCE_USE_FLASHINFER=0
  -e "VLLM_USE_RUST_FRONTEND=${RUST_FRONTEND:-0}"
  -e VLLM_PLUGINS=
  -e DG_JIT_USE_NVRTC=0
  -e MAX_JOBS=2
  -e NCCL_IB_ADDR_FAMILY=AF_INET
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1
  -e HF_HOME=/cache/huggingface -e XDG_CACHE_HOME=/cache
  -e VLLM_CACHE_ROOT=/cache/vllm -e TRITON_CACHE_DIR=/cache/triton
  -e FLASHINFER_WORKSPACE_BASE=/opt/flashinfer-cache
  -e TORCH_CUDA_ARCH_LIST=12.1a -e FLASHINFER_CUDA_ARCH_LIST=12.1a
  -e FLASHINFER_DISABLE_VERSION_CHECK=1 -e VLLM_HAS_FLASHINFER_CUBIN=1
  -e FLASHINFER_NVCC_THREADS=1 -e VLLM_USE_BREAKABLE_CUDAGRAPH=1
  -e TILELANG_CACHE_DIR=/cache/tilelang
)
for key in NCCL_CUMEM_ENABLE NCCL_NVLS_ENABLE PYTORCH_CUDA_ALLOC_CONF VLLM_USE_FLASHINFER_SAMPLER CUDA_LOG_FILE; do
  if [[ -n ${!key:-} ]]; then common+=(-e "$key=${!key}"); fi
done

if [[ "$action" == preflight ]]; then
  uname -a
  cat /etc/os-release
  nvidia-smi
  free -h
  df -h "$MODEL_STORE" "$RUN_DIR"
  ip -br addr show dev "$FABRIC_IF"
  if command -v ibdev2netdev >/dev/null; then ibdev2netdev; fi
  docker run --rm "${common[@]}" --entrypoint python3 "$IMAGE" /kit/preflight.py "$model_path"
  exit 0
fi
if [[ "$action" == nccl ]]; then
  test_nodes=${NCCL_TEST_NODES:-8}
  [[ "$test_nodes" =~ ^[248]$ && "$NODE_RANK" -lt "$test_nodes" ]] || { echo 'Use ranks 0..N-1 for N=2,4,8' >&2; exit 2; }
  exec docker run --rm --name "${CONTAINER_PREFIX:-deepseek-v41}-nccl-${test_nodes}-r${NODE_RANK}" "${common[@]}" --entrypoint torchrun "$IMAGE" \
    --nnodes "$test_nodes" --nproc-per-node 1 --node-rank "$NODE_RANK" \
    --master-addr "$HEAD_IP" --master-port "${NCCL_TEST_PORT:-29511}" \
    /kit/nccl_smoke.py
fi

engram_offload=${ENGRAM_CPU_OFFLOAD:-false}
[[ "$engram_offload" == true || "$engram_offload" == false ]] || { echo 'ENGRAM_CPU_OFFLOAD must be true or false' >&2; exit 2; }
vllm_args=(
  serve "$model_path" --served-model-name deepseek-v41-flash
  --distributed-executor-backend mp
  --tensor-parallel-size 8 --pipeline-parallel-size 1 --data-parallel-size 1
  --nnodes 8 --node-rank "$NODE_RANK" --master-addr "$HEAD_IP"
  --master-port "${MASTER_PORT:-29501}"
  --dtype bfloat16
  --default-chat-template-kwargs '{"thinking":false}'
  --tool-call-parser deepseek_v41 --enable-auto-tool-choice
  --reasoning-parser deepseek_v41
  --attention-backend FLASHINFER_MLA_SPARSE_DSV41
  --kv-cache-dtype "${KV_CACHE_DTYPE:-fp8_ds_mla}" --block-size 128
  --engram-config "{\"cpu_offload\":$engram_offload}"
  --max-model-len "${MAX_MODEL_LEN:-8192}"
  --max-num-batched-tokens "${MAX_BATCH_TOKENS:-1024}"
  --max-num-seqs "${MAX_NUM_SEQS:-1}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.75}"
  --enable-chunked-prefill
  --disable-custom-all-reduce
  --load-format safetensors --safetensors-load-strategy lazy
)
if [[ ${ENABLE_VISION:-0} == 1 ]]; then
  vllm_args+=(--limit-mm-per-prompt '{"image":4}' --mm-processor-cache-gb 1)
else
  vllm_args+=(--language-model-only)
fi
spec_method=${SPEC_METHOD:-none}
case "$spec_method" in
  none) ;;
  dspark)
    spec_k=${SPEC_K:-5}
    [[ "$spec_k" =~ ^[1-9][0-9]*$ ]] || { echo 'SPEC_K must be a positive integer' >&2; exit 2; }
    vllm_args+=(--speculative-config "{\"method\":\"dspark\",\"num_speculative_tokens\":$spec_k,\"draft_sample_method\":\"probabilistic\",\"rejection_sample_method\":\"block\",\"enable_adaptive_verification\":false}")
    ;;
  *) echo 'SPEC_METHOD must be none or dspark' >&2; exit 2 ;;
esac
if [[ ${ENABLE_GRAPHS:-0} == 1 ]]; then
  # Exact uniform target/draft batches avoid padding the SM121 sparse kernels.
  graph_sizes=$(python3 - "$spec_method" "${SPEC_K:-5}" "${MAX_NUM_SEQS:-1}" <<'PY'
import json, sys
method, k, seqs = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
assert 1 <= seqs <= 64
sizes = set()
if method == 'dspark':
    sizes.update(i * step for i in range(1, seqs + 1) for step in (k, k + 1))
else:
    sizes.update(range(1, seqs + 1))
print(json.dumps(sorted(sizes), separators=(',', ':')))
PY
)
  vllm_args+=(--compilation-config "{\"cudagraph_mode\":\"FULL_AND_PIECEWISE\",\"cudagraph_capture_sizes\":$graph_sizes}")
else
  vllm_args+=(--enforce-eager)
fi
if [[ ${ENABLE_PREFIX_CACHE:-0} == 1 ]]; then
  vllm_args+=(--enable-prefix-caching)
else
  vllm_args+=(--no-enable-prefix-caching)
fi
if [[ "$NODE_RANK" == 0 ]]; then
  vllm_args+=(--host "${API_HOST:-127.0.0.1}" --port "${API_PORT:-8000}")
else
  vllm_args+=(--headless)
fi
if [[ "$action" == dry-run ]]; then
  printf '%q ' docker run -d --name "$name" "${common[@]}" --entrypoint vllm "$IMAGE" "${vllm_args[@]}"
  printf '\n'
  exit 0
fi
if [[ "$action" == validate-config ]]; then
  exec docker run --rm "${common[@]}" --entrypoint python3 "$IMAGE" /kit/validate-config.py "${vllm_args[@]:1}"
fi
echo "Starting $name with the candidate image."
exec docker run -d --name "$name" "${common[@]}" --entrypoint vllm "$IMAGE" "${vllm_args[@]}"
