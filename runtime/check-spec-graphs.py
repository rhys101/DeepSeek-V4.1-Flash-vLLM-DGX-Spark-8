"""Check exact DSpark/target sparse-attention shapes in eager and CUDA graphs."""
import json
import torch
from vllm.models.deepseek_v4.common.ops import quantize_and_insert_k_cache
from vllm.utils.flashinfer import flashinfer_trtllm_batch_decode_sparse_mla_dsv4 as attend

torch.set_num_threads(2)
torch.manual_seed(41)
assert torch.cuda.get_device_capability() == (12, 1)

def make_cache(page):
    kv = torch.randn(2048, 512, dtype=torch.bfloat16, device='cuda')
    packed = torch.zeros(2048 // page, page * 584, dtype=torch.uint8, device='cuda')
    quantize_and_insert_k_cache(kv, packed, torch.arange(2048, device='cuda'), block_size=page)
    return packed.view(2048 // page, page, 584).unsqueeze(1)

swa = make_cache(64)
workspace = torch.empty(128 * 2048 * 2048, dtype=torch.uint8, device='cuda')
sink = torch.linspace(-1, 1, 8, device='cuda')
for n, width, valid in [(5, 192, 133), (6, 128, 128), (40, 192, 133), (48, 128, 128), (5, 1216, 1029), (6, 1152, 1024), (40, 1216, 1029), (48, 1152, 1024)]:
    q = torch.randn(n, 8, 512, dtype=torch.bfloat16, device='cuda')
    indices = torch.full((n, width), -1, device='cuda', dtype=torch.int32)
    indices[:, :valid] = torch.arange(valid, device='cuda', dtype=torch.int32)
    lengths = torch.full((n,), valid, dtype=torch.int32, device='cuda')
    for extra_page in [0, 64, 128]:
        kwargs = {}
        if extra_page:
            kwargs = dict(compressed_kv_cache=make_cache(extra_page),
                          extra_sparse_indices=torch.arange(512, device='cuda', dtype=torch.int32).expand(n, -1).contiguous(),
                          extra_sparse_topk_lens=torch.full_like(lengths, 512))
        def run():
            return attend(query=q, swa_kv_cache=swa, workspace_buffer=workspace,
                          sparse_indices=indices, swa_topk_lens=lengths,
                          bmm1_scale=512 ** -.5, sinks=sink, kv_layout='HND', **kwargs)
        for _ in range(3):
            expected = run()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = run()
        for scale in [1.0, -0.5, 2.0]:
            q.mul_(scale)
            expected = run()
            graph.replay()
            torch.cuda.synchronize()
            assert torch.isfinite(actual).all().item()
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        print(json.dumps(dict(status='PASS', tokens=n, swa_width=width, extra_page=extra_page,
                              graph_replays=3, comparison='bitwise_eager_with_changed_query')), flush=True)
        del graph, actual, expected
print('PASS: exact DSpark/target sparse-attention CUDA graph replay', flush=True)
