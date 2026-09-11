"""Native/padded MLA numerical and graph checks on the installed Spark stack."""
import ast
import json
from pathlib import Path
import statistics
from types import SimpleNamespace
import unittest

import torch

import test_flash_mla_backends as upstream
from sglang.kernels.ops.attention import flash_mla_sm120 as fmod

assert torch.cuda.get_device_capability() in ((12, 0), (12, 1))
fmod._sm120_default_backend = 'flashinfer'
result = unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite([
    upstream.TestEntryPointDispatch('test_flashinfer_exact_heads_match_padded_64_heads')
]))
assert result.wasSuccessful() and not result.skipped, result

# Exercise the backported selection and sink-view methods without constructing
# a model or loading checkpoint weights. Compile their unmodified AST bodies.
path = Path('/sgl-workspace/sglang/python/sglang/srt/models/deepseek_v4.py')
tree = ast.parse(path.read_text())
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MqaAttentionBase')
methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in ('_kernel_num_heads', '_local_attn_sink')]
assert len(methods) == 2
env = SimpleNamespace(SGLANG_SM120_FLASHMLA_BACKEND=SimpleNamespace(get=lambda: 'flashinfer'))
ns = dict(torch=torch, Optional=__import__('typing').Optional, envs=env,
          get_platform=lambda: SimpleNamespace(is_sm120=True), SM120_DECODE_MAX_TOKENS=64)
exec(compile(ast.fix_missing_locations(ast.Module(body=methods, type_ignores=[])), str(path), 'exec'), ns)
obj = SimpleNamespace(attn_tp_size=8, n_local_heads=8, n_heads=64, attn_tp_rank=2,
                      _attn_sink_local=None, attn_sink=torch.arange(64, device='cuda', dtype=torch.float32))
for tokens in (1, 6, 24, 48, 64, 65):
    assert ns['_kernel_num_heads'](obj, tokens) == 8
small = ns['_local_attn_sink'](obj, 8)
wide = ns['_local_attn_sink'](obj)
assert small.data_ptr() == wide.data_ptr() and wide.shape == (64,)
torch.testing.assert_close(small, obj.attn_sink[16:24], rtol=0, atol=0)
assert not torch.count_nonzero(wide[8:]).item()
env.SGLANG_SM120_FLASHMLA_BACKEND.get = lambda: 'triton'
assert ns['_kernel_num_heads'](obj, 6) == 64
assert ns['_kernel_num_heads'](obj, 65) == 8
print('MODEL_HEAD_SELECTION_AND_SINK_VIEWS PASS', flush=True)

def graph_call(q, sink, common):
    for _ in range(3):
        fmod.flash_mla_with_kvcache_sm120(q=q, attn_sink=sink, **common)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out, _ = fmod.flash_mla_with_kvcache_sm120(q=q, attn_sink=sink, **common)
    graph.replay()
    torch.cuda.synchronize()
    return graph, out

def time_graph(graph):
    samples=[]
    for _ in range(15):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(16): graph.replay()
        end.record(); end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / 16)
    return statistics.median(samples)

device=torch.device('cuda')
rows=[]
with torch.inference_mode():
    for tokens in (1, 6, 24, 48):
        for dual in (False, True):
            heads, pages, page_size, topk = 8, 8, 64, 512
            cache,_=upstream._build_kvcache(pages,page_size,device=device,seed=17)
            extra,_=upstream._build_kvcache(pages,page_size,device=device,seed=23)
            q,indices=upstream._build_q_indices(tokens,heads,topk,pages,page_size,device=device,seed=29)
            lengths=torch.full((tokens,),topk,dtype=torch.int32,device=device)
            sink=torch.linspace(-1,1,heads,dtype=torch.float32,device=device)
            padded=q.new_zeros(tokens,1,64,upstream._D);padded[:,:,:heads].copy_(q)
            padded_sink=sink.new_zeros(64);padded_sink[:heads].copy_(sink)
            common=dict(k_cache=cache,indices=indices,topk_length=lengths,head_dim_v=upstream._D,
                        softmax_scale=upstream._D**-0.5,extra_k_cache=extra if dual else None,
                        extra_indices_in_kvcache=indices if dual else None,
                        extra_topk_length=lengths if dual else None)
            native_graph,native_out=graph_call(q,sink,common)
            padded_graph,padded_out=graph_call(padded,padded_sink,common)
            max_abs=0.0
            for shift in (0.0,0.03125,-0.0625):
                q.add_(shift);padded[:,:,:heads].copy_(q)
                native_graph.replay();padded_graph.replay();torch.cuda.synchronize()
                ref=padded_out[:,:,:heads].float();got=native_out.float()
                torch.testing.assert_close(got,ref,atol=5e-2,rtol=5e-2)
                max_abs=max(max_abs,float((got-ref).abs().max()))
            # Reverse the timing order between cases to reduce ordering bias.
            if dual:
                padded_us=time_graph(padded_graph);native_us=time_graph(native_graph)
            else:
                native_us=time_graph(native_graph);padded_us=time_graph(padded_graph)
            row=dict(tokens=tokens,heads=heads,dual_cache=dual,graph_replays_with_changed_input=3,
                     max_abs_error=max_abs,native_median_us=native_us,padded_median_us=padded_us,
                     kernel_speedup=padded_us/native_us)
            rows.append(row);print('NATIVE_HEAD_RESULT '+json.dumps(row),flush=True)
            del native_graph,padded_graph,native_out,padded_out
print('NATIVE_HEADS_PASS '+json.dumps(dict(device=str(torch.cuda.get_device_name()),
      capability=torch.cuda.get_device_capability(),upstream_numerical_subcases=12,
      model_selection_and_sink_views='PASS',rows=rows)),flush=True)
