"""Compare vision prefill and decode with FP32 attention over the packed KV.

The 0.05/0.05 threshold matches the pinned upstream DSv4 FP8 tests:
https://github.com/flashinfer-ai/flashinfer/blob/07869c61ba581e6d6b8ad8d142f4a6c89b707cc1/tests/attention/test_sparse_mla_sm120.py
"""
import json
import torch
from vllm.models.deepseek_v4.common.ops import quantize_and_insert_k_cache
from vllm.utils.flashinfer import flashinfer_trtllm_batch_decode_sparse_mla_dsv4 as attend
from vllm.models.deepseek_v4_1.nvidia.flashinfer_sparse import DeepseekV4FlashInferSM120Attention

torch.set_num_threads(2)
torch.manual_seed(416)
torch.backends.cuda.matmul.allow_tf32=False
assert torch.cuda.get_device_capability() == (12,1)
for ratio in (1,2):
    physical=DeepseekV4FlashInferSM120Attention.backend_cls.get_compressed_block_size(ratio)//ratio
    assert physical==64,(ratio,physical)
print('PASS: both candidate compression ratios select 64-state pages',flush=True)

def make_cache(page):
    kv=torch.randn(2048,512,device='cuda',dtype=torch.bfloat16)
    packed=torch.zeros(2048//page,page*584,device='cuda',dtype=torch.uint8)
    quantize_and_insert_k_cache(kv,packed,torch.arange(2048,device='cuda'),block_size=page)
    return packed.view(2048//page,page,584).unsqueeze(1)

def unpack(cache):
    blocks,_,page,_=cache.shape
    raw=cache.reshape(blocks,page*584)
    body=raw[:,:page*576].reshape(blocks,page,576)
    scales=raw[:,page*576:].reshape(blocks,page,8)[...,:7].float()
    values=body[...,:448].contiguous().view(torch.float8_e4m3fn).float().reshape(blocks,page,7,64)
    values=values*torch.pow(2.,scales-127.)[...,None]
    rope=body[...,448:].contiguous().view(torch.bfloat16).float()
    return torch.cat((values.reshape(blocks,page,448),rope),dim=-1).reshape(-1,512)

swa=make_cache(64)
workspace=torch.empty(128*1024*1024,device='cuda',dtype=torch.uint8)
sinks=torch.linspace(-1,1,8,device='cuda')
for n,page in [(80,0),(1024,64),(8192,64)]:
    q=torch.randn(n,8,512,device='cuda',dtype=torch.bfloat16)
    indices=torch.full((n,1152),-1,device='cuda',dtype=torch.int32)
    indices[:,:1024]=torch.arange(1024,device='cuda',dtype=torch.int32)
    lengths=torch.full((n,),1024,device='cuda',dtype=torch.int32)
    extra=make_cache(page) if page else None
    extra_indices=torch.arange(512,device='cuda',dtype=torch.int32).expand(n,-1).contiguous()
    extra_lengths=torch.full((n,),512,device='cuda',dtype=torch.int32)
    def call(rows):
        kwargs={}
        if page:
            kwargs=dict(compressed_kv_cache=extra,extra_sparse_indices=extra_indices[:rows],extra_sparse_topk_lens=extra_lengths[:rows])
        return attend(query=q[:rows],swa_kv_cache=swa,workspace_buffer=workspace,
            sparse_indices=indices[:rows],swa_topk_lens=lengths[:rows],
            bmm1_scale=512**-.5,sinks=sinks,kv_layout='HND',**kwargs)
    full=call(n)
    short=call(16)
    keys=unpack(swa)[:1024]
    if page:keys=torch.cat((keys,unpack(extra)[:512]),dim=0)
    scores=torch.matmul(q[:16].float(),keys.T)*(512**-.5)
    with_sink=torch.cat((scores,sinks[None,:,None].expand(16,8,1)),dim=-1)
    probabilities=torch.softmax(with_sink,dim=-1)[...,:keys.shape[0]]
    reference=torch.matmul(probabilities,keys)
    torch.cuda.synchronize()
    assert torch.isfinite(full).all().item()
    torch.testing.assert_close(full[:16].float(),reference,atol=.05,rtol=.05)
    torch.testing.assert_close(short.float(),reference,atol=.05,rtol=.05)
    difference=(full[:16].float()-short.float()).abs().max().item()
    print(json.dumps(dict(status='PASS',tokens=n,vision_index_width=1152,extra_page=page,
        compared_query_rows=16,max_abs_prefill_decode_difference=difference,
        max_abs_prefill_reference_difference=(full[:16].float()-reference).abs().max().item(),
        max_abs_decode_reference_difference=(short.float()-reference).abs().max().item(),
        prefill_relative_rms=((full[:16].float()-reference).square().mean()/reference.square().mean()).sqrt().item(),
        atol=.05,rtol=.05)),flush=True)
