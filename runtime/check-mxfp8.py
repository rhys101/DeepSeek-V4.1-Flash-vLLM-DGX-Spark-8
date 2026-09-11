"""Compare existing FlashInfer CUTLASS and b12x on TP8 model dimensions.

Run only with the serving model stopped. FP32 matmul of independently
dequantized operands is the numerical reference; time CUDA graph replay.
"""
import argparse
import gc
import importlib.metadata
import json
import pathlib
import statistics
import time

import torch
from flashinfer import mm_mxfp8
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    mxfp8_e4m3_quantize, swizzle_mxfp8_scale,
)

torch.cuda.set_device(0)
torch.manual_seed(20260911)
torch.backends.cuda.matmul.allow_tf32 = False
parser=argparse.ArgumentParser()
parser.add_argument('--out', type=pathlib.Path, default=pathlib.Path('mxfp8-results.json'))
OUT=parser.parse_args().out
OUT.parent.mkdir(parents=True,exist_ok=True)
rows = []
metadata = dict(device=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability(), torch=torch.__version__, flashinfer=importlib.metadata.version('flashinfer-python'), dsl=importlib.metadata.version('nvidia-cutlass-dsl'))
print(json.dumps(metadata), flush=True)

def quant(x):
    q, scales = mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=False)
    scales = scales[:x.shape[0], :x.shape[1]//32].contiguous()
    deq = (q.float().reshape(x.shape[0], -1, 32) * torch.exp2(scales.float()-127).unsqueeze(-1)).reshape(x.shape)
    swizzled = swizzle_mxfp8_scale(scales, *x.shape)
    return q, swizzled, deq

def error(output, reference):
    diff = output.float() - reference
    return dict(finite=bool(torch.isfinite(output).all()), relative_l2=float(diff.norm()/reference.norm().clamp_min(1e-12)), max_abs=float(diff.abs().max()))

for label, n, k in [('fused_qkv',1792,5120), ('q_b_and_indexer',4096,1280), ('o_b',5120,1024), ('engram_replicated',25600,6144)]:
    for m in [1,5,6,40,48,128,256,1024]:
        started=time.monotonic()
        a,sa,ar=quant(torch.randn((m,k),device='cuda',dtype=torch.bfloat16))
        b,sb,br=quant(torch.randn((n,k),device='cuda',dtype=torch.bfloat16))
        ref=ar @ br.t()
        row=dict(layer=label,m=m,n=n,k=k,backends={})
        for backend in ['cutlass','b12x']:
            def fn():
                return mm_mxfp8(a,b.t(),sa,sb,out_dtype=torch.bfloat16,use_8x4_sf_layout=False,backend=backend)
            output=fn()
            err=error(output,ref)
            assert err['finite'] and err['relative_l2'] < 0.008, (row,backend,err)
            for _ in range(3):fn()
            torch.cuda.synchronize()
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(20):captured=fn()
            times=[]
            for _ in range(7):
                start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                start.record();graph.replay();end.record();end.synchronize()
                times.append(start.elapsed_time(end)*1000/20)
            # Replay with different activation values AND scales at the same addresses.
            a2,sa2,ar2=quant(torch.randn((m,k),device='cuda',dtype=torch.bfloat16)*3.5)
            a_saved=a.clone();sa_saved=sa.clone()
            a.copy_(a2);sa.copy_(sa2)
            graph.replay();torch.cuda.synchronize()
            changed=error(captured,ar2 @ br.t())
            assert changed['finite'] and changed['relative_l2'] < 0.008,(row,backend,changed)
            a.copy_(a_saved);sa.copy_(sa_saved)
            row['backends'][backend]=dict(graph_median_us=statistics.median(times),graph_samples_us=times,error=err,changed_input_graph_error=changed)
            del graph,captured,output,a2,sa2,ar2,a_saved,sa_saved
        row['speedup']=row['backends']['cutlass']['graph_median_us']/row['backends']['b12x']['graph_median_us']
        row['elapsed_seconds']=time.monotonic()-started
        rows.append(row)
        OUT.write_text(json.dumps(dict(metadata=metadata,rows=rows,status='running'),indent=2)+'\n')
        print(json.dumps(row),flush=True)
        del a,b,sa,sb,ar,br,ref
        gc.collect()
OUT.write_text(json.dumps(dict(metadata=metadata,rows=rows,status='PASS'),indent=2)+'\n')
print('PASS: all backends, shapes, and changed-input graph replays',flush=True)
