"""GPU integration check for the installed vLLM small-M routing patch."""
import argparse
import json
import os
import pathlib

import torch
from torch.nn import Parameter
from vllm.model_executor.kernels.linear.mxfp8.flashinfer import FlashInferCutlassMxfp8LinearKernel
from vllm.model_executor.kernels.linear.mxfp8.Mxfp8LinearKernel import Mxfp8LinearLayerConfig
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import mxfp8_e4m3_quantize
from vllm.utils import flashinfer as vf

parser=argparse.ArgumentParser()
parser.add_argument('--out',type=pathlib.Path,default=pathlib.Path('mxfp8-routing-validation.json'))
output_path=parser.parse_args().out
output_path.parent.mkdir(parents=True,exist_ok=True)
os.environ['SPARK_MXFP8_B12X_MAX_M']='128'
torch.manual_seed(91)
kernel=FlashInferCutlassMxfp8LinearKernel(Mxfp8LinearLayerConfig())
n,k=1792,5120
layer=torch.nn.Module()
w,ws=mxfp8_e4m3_quantize(torch.randn((n,k),device='cuda',dtype=torch.bfloat16),is_sf_swizzled_layout=False)
layer.weight=Parameter(w,requires_grad=False)
layer.weight_scale=Parameter(ws,requires_grad=False)
kernel.process_weights_after_loading(layer)
assert kernel._spark_b12x_device and kernel._spark_b12x_max_m==128
real=vf.mm_mxfp8
calls=[]
def observe(*args,**kwargs):
    calls.append(kwargs['backend'])
    return real(*args,**kwargs)
vf.mm_mxfp8=observe
rows=[]
try:
    for m in [1,5,6,40,48,128,129,256]:
        x=torch.randn((m,k),device='cuda',dtype=torch.bfloat16)
        bias=torch.randn((n,),device='cuda',dtype=torch.bfloat16)
        calls.clear()
        y=kernel.apply_weights(layer,x,bias)
        expected='b12x' if m<=128 else 'cutlass'
        assert calls==[expected],(m,calls)
        aq,sa=mxfp8_e4m3_quantize(x,is_sf_swizzled_layout=True)
        ref=real(aq,layer.weight.t(),sa,layer.weight_scale,out_dtype=torch.bfloat16,backend='cutlass')+bias
        torch.testing.assert_close(y,ref,rtol=0.01,atol=0.0625)
        for _ in range(2):kernel.apply_weights(layer,x,bias)
        torch.cuda.synchronize()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):captured=kernel.apply_weights(layer,x,bias)
        x.copy_(torch.randn_like(x)*2)
        graph.replay();torch.cuda.synchronize()
        aq,sa=mxfp8_e4m3_quantize(x,is_sf_swizzled_layout=True)
        ref=real(aq,layer.weight.t(),sa,layer.weight_scale,out_dtype=torch.bfloat16,backend='cutlass')+bias
        torch.testing.assert_close(captured,ref,rtol=0.01,atol=0.0625)
        rows.append(dict(m=m,backend=expected,finite=bool(torch.isfinite(captured).all()),changed_input_graph='PASS'))
        del graph,captured
    shaped=torch.randn((2,3,k),device='cuda',dtype=torch.bfloat16)
    assert kernel.apply_weights(layer,shaped).shape==(2,3,n)
finally:
    vf.mm_mxfp8=real
result=dict(status='PASS',device=torch.cuda.get_device_name(),rows=rows,three_dimensional_input='PASS')
output_path.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
