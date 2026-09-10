#!/usr/bin/env bash
# The reference's stable-only CMake build, with bounded parallelism and pins.
set -Eeuo pipefail
export TORCH_CUDA_ARCH_LIST=12.1a
cd /src
pip install --disable-pip-version-check cmake==3.31.6
mkdir -p build/_deps/cutlass-src
curl -fSL --retry 3 --max-time 900 \
  https://codeload.github.com/NVIDIA/cutlass/tar.gz/cb4247394dd82148787aed73e5dc7cef33cbf862 \
  | tar xz -C build/_deps/cutlass-src --strip-components=1
test -f build/_deps/cutlass-src/include/cutlass/cutlass.h
cp CMakeLists.txt CMakeLists.txt.original
python3 - <<'PY'
from pathlib import Path
import re
p=Path('CMakeLists.txt')
s,n=re.subn(r'^(\s*)include\(cmake/external_projects/',r'\1# STABLE-ONLY: include(cmake/external_projects/',p.read_text(),flags=re.M)
assert n>0
p.write_text(s)
print('Disabled external extension projects:',n,flush=True)
PY
python_path=$(python3 -c 'import sys; print(":".join(p for p in sys.path if p))')
torch_prefix=$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)')
nvrtc=/usr/local/cuda/lib64/libnvrtc.so.13
test -f "$nvrtc"
cmake -S /src -B /src/build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DVLLM_TARGET_DEVICE=cuda -DVLLM_PYTHON_EXECUTABLE=/usr/bin/python3 \
  -DVLLM_PYTHON_PATH="$python_path" \
  -DFETCHCONTENT_BASE_DIR=/src/build/_deps \
  -DFETCHCONTENT_SOURCE_DIR_CUTLASS=/src/build/_deps/cutlass-src \
  -DCMAKE_PREFIX_PATH="$torch_prefix" -DNVCC_THREADS=2 \
  -DCUDA_nvrtc_LIBRARY="$nvrtc"
cmake --build /src/build --target _C_stable_libtorch -j 8
test -s /src/build/_C_stable_libtorch.abi3.so
sha256sum /src/build/_C_stable_libtorch.abi3.so > /src/build/stable-extension.sha256
nvcc --version > /src/build/nvcc-version.txt
printf 'STABLE_EXTENSION_BUILD_PASS\n'
