"""Check the installed SGLang dense hook against independently dequantized FP32 operands.

Run in the candidate image on an idle GPU. No throughput claims from this check.
"""
import gc
import json
from pathlib import Path
import torch
from flashinfer import mxfp8_quantize
import sglang.srt.layers.quantization.fp8_utils as fp8

torch.manual_seed(20260911)
torch.backends.cuda.matmul.allow_tf32=False
assert torch.cuda.get_device_capability()==(12,1)
assert fp8.flashinfer_mxfp8_blockscaled_linear.__module__=='mxfp8_b12x', 'Candidate hook not installed'
rows=[]
original_mm=fp8.flashinfer_mm_mxfp8
calls=[]
def observe(*args,**kwargs):
    calls.append(kwargs.get('backend'));return original_mm(*args,**kwargs)
fp8.flashinfer_mm_mxfp8=observe

def quant(x):
    q,s=mxfp8_quantize(x,is_sf_swizzled_layout=False,alignment=32)
    q2,sw=mxfp8_quantize(x,is_sf_swizzled_layout=True,alignment=32)
    assert torch.equal(q.view(torch.uint8),q2.view(torch.uint8))
    # FlashInfer 0.6.18 returns layout_linear as a flat row-major buffer.
    groups=x.shape[1]//32
    assert s.numel()>=x.shape[0]*groups and s.numel()%groups==0
    s=s.reshape(-1,groups)[:x.shape[0]].contiguous()
    exp=s.to(torch.int32)
    scale=(exp<<23).view(torch.float32)
    scale=torch.where(exp==0,2.0**-127,scale)
    deq=(q.float().view(x.shape[0],-1,32)*scale.unsqueeze(-1)).reshape(x.shape)
    return q2,sw.reshape(-1),deq

def check(output,reference):
    diff=output.float()-reference
    e=dict(finite=bool(torch.isfinite(output).all()),relative_l2=float(diff.norm()/reference.norm().clamp_min(1e-12)),max_abs=float(diff.abs().max()))
    assert e['finite'] and e['relative_l2']<0.008,e
    return e

try:
    for label,n,k in [('fused_qkv',1792,5120),('q_b_indexer',4096,1280),('o_b',5120,1024),('engram',25600,6144)]:
        w,ws,wr=quant(torch.randn(n,k,device='cuda',dtype=torch.bfloat16))
        for m in (1,5,6,40,48,128,256,1024):
            x=torch.randn(m,k,device='cuda',dtype=torch.bfloat16)
            _,_,xr=quant(x)
            def run():return fp8.flashinfer_mxfp8_blockscaled_linear(x,w,ws,output_dtype=torch.bfloat16,backend='cutlass')
            calls.clear();y=run();eager=check(y,xr@wr.t())
            for _ in range(3):run()
            torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):captured=run()
            x.copy_(torch.randn_like(x)*3.5)
            _,_,xr2=quant(x)
            g.replay();torch.cuda.synchronize()
            changed=check(captured,xr2@wr.t())
            row=dict(layer=label,m=m,n=n,k=k,backend_calls=sorted(set(calls)),eager=eager,changed_input_graph=changed)
            rows.append(row);print(json.dumps(row),flush=True)
            del g,captured,y,x,xr,xr2;gc.collect()
        del w,ws,wr;gc.collect()
    report=dict(status='PASS',torch=torch.__version__,cases=rows,reference='FP32 GEMM of independently decoded E4M3/E8M0 operands; relative L2 < 0.008')
    Path('/state/mxfp8-check.json').write_text(json.dumps(report,indent=2)+'\n')
    print('PASS: 32 installed SGLang-hook shapes and changed-input CUDA graph replays',flush=True)
finally:fp8.flashinfer_mm_mxfp8=original_mm
