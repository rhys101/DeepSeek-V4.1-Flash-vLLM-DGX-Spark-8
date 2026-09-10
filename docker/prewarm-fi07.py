"""Build the reference's two JIT modules under the exact serving environment."""
import os
import pathlib
import time

for key in ('FLASHINFER_JIT_VERBOSE', 'FLASHINFER_JIT_DEBUG', 'FLASHINFER_JIT_LINEINFO'):
    os.environ.pop(key, None)

from flashinfer.jit.gemm import gen_gemm_sm120_module_cutlass_mxfp8

t = time.monotonic()
spec = gen_gemm_sm120_module_cutlass_mxfp8()
spec.build()
compiled = spec.is_compiled
assert compiled() if callable(compiled) else compiled
print('MXFP8 COMPILED', time.monotonic()-t, flush=True)

from flashinfer.mla._sparse_mla_sm120 import get_sparse_mla_sm120_module

t = time.monotonic()
try:
    get_sparse_mla_sm120_module()
except Exception as exc:
    # Docker builds have no GPU. Compilation must still produce the shared
    # library; GPU loading and execution are checked separately after building.
    print('Sparse module load result:', type(exc).__name__, str(exc), flush=True)
libs = list(pathlib.Path(os.environ['FLASHINFER_WORKSPACE_BASE']).rglob('sparse_mla_sm120*.so'))
assert libs, 'Sparse MLA compilation produced no shared library'
print('SPARSE COMPILED', time.monotonic()-t, [str(p) for p in libs], flush=True)
