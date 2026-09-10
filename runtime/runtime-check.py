"""Cheap checks only: does not load model weights or certify kernel correctness."""
import importlib.metadata
import json
import pathlib
import platform
import struct
import sys

import torch

assert platform.machine() in ('aarch64', 'arm64'), platform.machine()
assert torch.cuda.is_available(), 'CUDA is unavailable'
assert torch.cuda.device_count() == 1, 'Expose exactly one GPU per Spark'
assert torch.cuda.get_device_capability(0) == (12, 1), 'Expected GB10 / SM121'
print('GPU:', torch.cuda.get_device_name(0))
print('CUDA runtime:', torch.version.cuda, 'NCCL:', torch.cuda.nccl.version())
print('Torch architectures:', torch.cuda.get_arch_list())
for package in ('vllm', 'torch', 'triton', 'flashinfer-python', 'flashinfer-cubin', 'transformers'):
    try:
        print(package, importlib.metadata.version(package))
    except importlib.metadata.PackageNotFoundError:
        print(package, 'not separately installed')

a = torch.ones((256, 256), device='cuda', dtype=torch.bfloat16)
b = a @ a
torch.cuda.synchronize()
assert torch.all(b == 256).item(), 'BF16 GEMM result is incorrect'
del a, b

from vllm.model_executor.models import ModelRegistry
assert 'DeepseekV41ForCausalLM' in ModelRegistry.get_supported_archs(), 'V4.1 architecture missing'
from vllm.models.deepseek_v4_1.nvidia.flashinfer_sparse import DeepseekV4FlashInferMLASparseBackend
from vllm.platforms.interface import DeviceCapability
assert DeepseekV4FlashInferMLASparseBackend.supports_compute_capability(DeviceCapability(12, 1))
from vllm.utils.flashinfer import has_flashinfer_sparse_mla_sm120_config
assert has_flashinfer_sparse_mla_sm120_config(8, 128), 'FlashInfer lacks the TP8 / SWA128 specialization'
assert has_flashinfer_sparse_mla_sm120_config(8, 1152), 'FlashInfer lacks the TP8 / vision specialization'
print('V4.1 registry, SM12x backend and FlashInfer TP8 text/vision dispatch are present.')


print('PASS: runtime and V4.1 backend availability checks; full model kernels still need validation.')
