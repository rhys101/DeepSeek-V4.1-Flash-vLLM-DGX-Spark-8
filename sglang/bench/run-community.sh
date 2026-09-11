#!/usr/bin/env bash
set -Eeuo pipefail
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
base=${1:?Usage: run-community.sh API_BASE_V1 UNIQUE_OUTPUT_DIR [label]}
output=${2:?Supply a new output directory}
label=${3:-ep4-matched}
[[ ! -e "$output" ]] || { echo 'Use a new output directory to preserve prior results.' >&2; exit 2; }
mkdir -p "$output"
python3 -u "$repo_dir/bench/v41bench.py" --base "$base" --model deepseek-v41-flash \
  --label "$label" --out "$output" --levels 1,2,3,4,5,6,8 --prefill 2000,8000,32000,64000 \
  --notes 'SGLang EP4, eight DGX Sparks, TP8/EP4, native resident Engram, DSpark 5, FlashInfer 0.6.18; attach resolved configuration and source identity' \
  | tee "$output/console.log"
