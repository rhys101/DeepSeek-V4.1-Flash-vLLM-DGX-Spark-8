#!/usr/bin/env bash
# Build on an idle Linux ARM64 Spark using the pinned reference base.
set -Eeuo pipefail
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
build_dir=${BUILD_DIR:-$repo_dir/.build}
image=${IMAGE_TAG:-deepseek-v41-spark8:2026-09-10-fi07-vision}
[[ $(uname -m) == aarch64 ]] || { echo 'Build on a Linux ARM64 Spark.' >&2; exit 2; }
mkdir -p "$build_dir"
exec 9>"$build_dir/build.lock"
flock -n 9 || { echo 'Another build is running.' >&2; exit 2; }
read_lock() {
  python3 - "$repo_dir/versions.lock.json" "$1" <<'LOCK'
import json,sys
x=json.load(open(sys.argv[1]))
for key in sys.argv[2].split('.'): x=x[key]
print(x)
LOCK
}
docker buildx build --load --progress=plain --platform linux/arm64   -f "$repo_dir/docker/Dockerfile"   --build-arg "BASE_IMAGE=$(read_lock base_image)"   --build-arg "VLLM_SHA=$(read_lock sources.vllm.commit)"   --build-arg "FI_SHA=$(read_lock sources.flashinfer.commit)"   --build-arg "CUTLASS_SHA=$(read_lock sources.flashinfer_cutlass.commit)"   --build-arg "CCCL_SHA=$(read_lock sources.flashinfer_cccl.commit)"   --build-arg "SPDLOG_SHA=$(read_lock sources.flashinfer_spdlog.commit)"   -t "$image" "$repo_dir" 2>&1 | tee "$build_dir/build.log"
docker image inspect --format '{{.Id}}' "$image" | tee "$build_dir/image-id.txt"
echo 'Build complete. Run the GPU, model and fabric checks before serving.'
