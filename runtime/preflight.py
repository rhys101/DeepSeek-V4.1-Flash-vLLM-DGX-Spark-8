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
print('V4.1 registry, SM12x backend and FlashInfer (8, 128) dispatch are present.')

root = pathlib.Path(sys.argv[1])
config = json.loads((root / 'config.json').read_text())
assert config['architectures'] == ['DeepseekV41ForCausalLM'], config['architectures']
print('Checkpoint quantization:', config.get('quantization_config'))
index = json.loads((root / 'model.safetensors.index.json').read_text())
shards = sorted(set(index['weight_map'].values()))
print('Checkpoint:', len(shards), 'shards;', index['metadata']['total_size'], 'tensor bytes')
actual_keys = {}
total_tensor_bytes = 0
for filename in shards:
    path = root / filename
    assert path.is_file(), f'Missing shard or broken snapshot symlink: {path}'
    with path.open('rb') as handle:
        size_raw = handle.read(8)
        assert len(size_raw) == 8, f'Truncated header: {path}'
        header_len = struct.unpack('<Q', size_raw)[0]
        assert 0 < header_len < 100_000_000, f'Invalid header length: {path}'
        header = json.loads(handle.read(header_len))
    end = 0
    for key, value in header.items():
        if key == '__metadata__':
            continue
        start, stop = value['data_offsets']
        assert 0 <= start <= stop, f'Invalid offsets: {path} {key}'
        end = max(end, stop)
        total_tensor_bytes += stop - start
        assert key not in actual_keys, f'Duplicate tensor: {key}'
        actual_keys[key] = filename
    assert path.stat().st_size == 8 + header_len + end, f'Truncated/unexpected file size: {path}'
assert actual_keys == index['weight_map'], 'Shard contents differ from the checkpoint index'
assert total_tensor_bytes == index['metadata']['total_size'], 'Tensor-byte total differs from index'
print('PASS: basic runtime, backend availability and checkpoint structure.')
print('Still required: checksum verification, NCCL transport tests, actual V4.1 kernel execution and output correctness.')
