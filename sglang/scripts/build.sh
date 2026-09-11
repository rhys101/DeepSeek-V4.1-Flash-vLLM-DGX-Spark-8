#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
config=${1:-"$root/configs/cluster.local.json"}
[[ -f "$config" ]] || { echo "Copy and edit configs/cluster.example.json first." >&2; exit 2; }
[[ $(uname -m) == aarch64 ]] || { echo 'Build on an idle Linux ARM64 Spark.' >&2; exit 2; }
python3 - "$root" "$config" <<'PY'
import pathlib,sys
root=pathlib.Path(sys.argv[1]);sys.path.insert(0,str(root/'scripts'))
from cluster import load,require_head,mapped,idle
c=load(sys.argv[2]);require_head(c);mapped(c,lambda n:idle(c,n))
PY
image=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["image"])' "$config")
mkdir -p "$root/.build"
docker buildx build --load --platform linux/arm64 --progress=plain -f "$root/docker/Dockerfile" -t "$image" "$root" 2>&1 | tee "$root/.build/build.log"
docker image inspect --format '{{.Id}}' "$image" > "$root/.build/image-id.txt"
python3 - "$root" "$config" <<'PY'
import json,pathlib,sys
root=pathlib.Path(sys.argv[1]);p=pathlib.Path(sys.argv[2]);d=json.loads(p.read_text())
d['expected_image_id']=(root/'.build/image-id.txt').read_text().strip()
t=p.with_suffix('.json.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)
print('Recorded built image:',d['expected_image_id'])
PY
