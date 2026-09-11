"""Idle-GPU check of TP8 row ownership, FP8 dequantization and CUDA graph replay."""
import importlib.metadata
import json
import torch
from sglang.kernels.ops.embeddings.engram_gather import engram_gather

torch.set_num_threads(2)
torch.manual_seed(41)
assert torch.cuda.get_device_capability() == (12, 1)
total, dim = 259, 256
weight = torch.randn(total, dim, device='cuda').to(torch.float8_e4m3fn)
scales = torch.randint(120, 135, (total, dim // 32), device='cuda', dtype=torch.uint8)
scales[0] = 0
exps = scales.to(torch.int32)
scale_float = (exps << 23).view(torch.float32)
scale_float = torch.where(exps == 0, 2.0 ** -127, scale_float)
reference = (weight.float().view(total, dim // 32, 32) * scale_float.unsqueeze(-1)).flatten(1).to(torch.bfloat16)
ids = torch.arange(48, device='cuda', dtype=torch.int64)
outputs, graphs = [], []
for rank in range(8):
    lo, hi = total * rank // 8, total * (rank + 1) // 8
    w, s = weight[lo:hi].clone(), scales[lo:hi].clone()
    out = torch.empty(ids.numel(), dim, device='cuda', dtype=torch.bfloat16)
    def run(w=w, s=s, out=out, lo=lo, hi=hi):
        engram_gather(w.data_ptr(), s.data_ptr(), ids, out, dim, 32, row_lo=lo, row_hi=hi)
    for _ in range(3): run()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): run()
    # Keep backing allocations alive for every replay.
    graphs.append((g, w, s)); outputs.append(out)
for offset in (0, 79, 211):
    ids.copy_((torch.arange(48, device='cuda', dtype=torch.int64) + offset) % total)
    for g, _, _ in graphs: g.replay()
    torch.cuda.synchronize()
    result = torch.stack(outputs).sum(0)
    torch.testing.assert_close(result, reference[ids], atol=0, rtol=0)
    assert torch.isfinite(result).all()
print(json.dumps(dict(status='PASS', test='native Engram gather, eight owned-row shards, three changed-index CUDA graph replays',
    comparison='bitwise FP32 dequantization rounded to BF16', cuda_capability=[12,1],
    torch=torch.__version__, flashinfer=importlib.metadata.version('flashinfer-python'))))
