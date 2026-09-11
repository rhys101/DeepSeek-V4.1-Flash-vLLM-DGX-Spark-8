"""Real-checkpoint EP4/MoE-TP2 component check. No serving code changes.

On one idle SM121, simulate each group's two TP ranks. Compare their sum
with the same group's two current EP8 ranks and with its unsplit expert set.
This tests local partition arithmetic, not network behavior or model quality.
"""
import contextlib, datetime, gc, hashlib, json, statistics, time
from pathlib import Path
import torch
from safetensors import safe_open
from flashinfer import block_scale_interleave, mxfp8_quantize
from flashinfer.fused_moe import cutlass_fused_moe
from flashinfer.fused_moe.core import ActivationType

OUT=Path('/evidence'); MODEL=Path('/models/DeepSeek-V4.1-Flash')
K,N=5120,2304
torch.backends.cuda.matmul.allow_tf32=False
RESULT=dict(status='RUNNING',started=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    gates=dict(relative_l2=0.02,cosine=0.9995),cases=[],timings=[],checkpoint_tensors=[],
    limitation='Same FlashInfer arithmetic and checkpoint. Synthetic inputs/routes, local sums; no full-model quality or serving performance claim.')
def save(phase,**kw):
    RESULT.update(phase=phase,**kw);(OUT/'result.json').write_text(json.dumps(RESULT,indent=2)+'\n')
    print(json.dumps(dict(phase=phase,**kw)),flush=True)
def metric(a,b):
    a,b=a.float(),b.float();assert torch.isfinite(a).all() and torch.isfinite(b).all()
    norm=float(b.norm())
    if norm==0:
        assert torch.count_nonzero(a)==0
        return dict(relative_l2=0,cosine=1,reference_norm=0)
    row=dict(relative_l2=float((a-b).norm())/norm,cosine=float(torch.nn.functional.cosine_similarity(a.flatten(),b.flatten(),dim=0)),reference_norm=norm)
    assert row['relative_l2']<=RESULT['gates']['relative_l2'] and row['cosine']>=RESULT['gates']['cosine'],row
    return row
def load_weights(prefix,group,total):
    E=total//4;idx=json.loads((MODEL/'model.safetensors.index.json').read_text())['weight_map']
    raw={k:torch.empty(s,dtype=torch.uint8,device='cuda') for k,s in dict(w13=(E,2*N,K//2),s13=(E,2*N,K//32),w2=(E,K,N//2),s2=(E,K,N//32)).items()}
    with contextlib.ExitStack() as stack:
        files={}
        for local in range(E):
            for proj,field,dest,part in [('w3','weight','w13',slice(0,N)),('w1','weight','w13',slice(N,2*N)),('w3','scale','s13',slice(0,N)),('w1','scale','s13',slice(N,2*N)),('w2','weight','w2',slice(None)),('w2','scale','s2',slice(None))]:
                key=f'{prefix}.ffn.experts.{group*E+local}.{proj}.{field}';fn=idx[key]
                if fn not in files:files[fn]=stack.enter_context(safe_open(str(MODEL/fn),framework='pt',device='cpu'))
                value=files[fn].get_tensor(key).contiguous().view(torch.uint8)
                assert value.shape==raw[dest][local,part].shape
                raw[dest][local,part].copy_(value)
                RESULT['checkpoint_tensors'].append(dict(key=key,file=fn,shape=list(value.shape),sha256=hashlib.sha256(value.numpy().tobytes()).hexdigest()))
    return raw
def prepare(raw):
    r=dict(raw)
    r['s13']=block_scale_interleave(raw['s13']).reshape_as(raw['s13'])
    r['s2']=block_scale_interleave(raw['s2']).reshape_as(raw['s2'])
    r['ones']=torch.ones(raw['w13'].shape[0],device='cuda',dtype=torch.float32)
    r['limits']=r['ones']*10
    return r
def split(raw,t):
    h=N//2;s=t*h
    return dict(w13=torch.cat([raw['w13'][:,s:s+h],raw['w13'][:,N+s:N+s+h]],dim=1).contiguous(),
        s13=torch.cat([raw['s13'][:,s:s+h],raw['s13'][:,N+s:N+s+h]],dim=1).contiguous(),
        w2=raw['w2'][:,:,s//2:(s+h)//2].contiguous(),s2=raw['s2'][:,:,s//32:(s+h)//32].contiguous())
def inputs(m,total,topk,group,mode,scale,seed):
    gen=torch.Generator(device='cuda').manual_seed(seed);E=total//4
    x=(torch.randn(m,K,device='cuda',generator=gen)*scale).bfloat16()
    scores=torch.randn(m,total,device='cuda',generator=gen)
    if mode=='local':
        scores[:,:group*E]=-float('inf');scores[:,(group+1)*E:]=-float('inf')
    elif mode=='nonlocal':scores[:,group*E:(group+1)*E]=-float('inf')
    v,ids=torch.topk(scores,topk,dim=1)
    return x,ids.int(),torch.softmax(v,dim=1).float()*1.5
def check_shape(prefix,group,total,topk,m,arms):
    x,ids,routes=inputs(m,total,topk,group,'mixed',2,800+m)
    graphs={};outputs={}
    for name,(raw,tp,tr,ep,er) in arms.items():
        out=torch.empty_like(x);outputs[name]=out
        def invoke(raw=raw,tp=tp,tr=tr,ep=ep,er=er,out=out):
            xq,xsf=mxfp8_quantize(x,is_sf_swizzled_layout=True,alignment=32)
            cutlass_fused_moe(input=xq,token_selected_experts=ids,token_final_scales=routes,
                fc1_expert_weights=raw['w13'].view(torch.int64),fc2_expert_weights=raw['w2'].view(torch.int64),
                output_dtype=torch.bfloat16,quant_scales=[raw['s13'].view(torch.int32),raw['ones'],raw['s2'].view(torch.int32),raw['ones']],
                input_sf=xsf,swiglu_limit=raw['limits'],tp_size=tp,tp_rank=tr,ep_size=ep,ep_rank=er,
                use_w4_group_scaling=False,use_mxfp8_act_scaling=True,activation_type=ActivationType.Swiglu,
                tune_max_num_tokens=1<<(m-1).bit_length(),output=out,use_fused_finalize=False)
        invoke();torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):invoke()
        graphs[name]=g
    addresses=[a.data_ptr() for a in (x,ids,routes,*outputs.values())]
    for j,(mode,scale) in enumerate([('mixed',2),('local',8),('nonlocal',2)]):
        for dst,src in zip((x,ids,routes),inputs(m,total,topk,group,mode,scale,1800+m+j)):dst.copy_(src)
        for name,g in graphs.items():outputs[name].fill_(float('nan'));g.replay()
        torch.cuda.synchronize()
        candidate=outputs['tp0'].float()+outputs['tp1'].float()
        baseline=outputs['ep0'].float()+outputs['ep1'].float()
        row=dict(prefix=prefix,group=group,tokens=m,mode=mode,scale=scale,
            candidate_vs_ep8=metric(candidate,baseline),candidate_vs_unsplit=metric(candidate,outputs['full']),baseline_vs_unsplit=metric(baseline,outputs['full']))
        before=torch.cuda.memory_allocated();copies={k:v.clone() for k,v in outputs.items()};with_copies=torch.cuda.memory_allocated()
        for name,g in graphs.items():g.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_allocated()==with_copies
        for name in outputs:assert torch.equal(copies[name],outputs[name]),name
        assert addresses==[a.data_ptr() for a in (x,ids,routes,*outputs.values())]
        row.update(replay_exact=True,replay_allocation_growth_bytes=0);RESULT['cases'].append(row)
        del copies
    # Avoid cache-hot-only predictions. Flush L2 before each event-timed replay.
    if m in (5,6,24,48):
        for dst,src in zip((x,ids,routes),inputs(m,total,topk,group,'mixed',2,2700+m)):dst.copy_(src)
        flush=torch.empty(64*1024*1024,dtype=torch.uint8,device='cuda');samples={k:[] for k in ('tp0','tp1','ep0','ep1')}
        names=list(samples)
        for j in range(15):
            for name in (names if j%2==0 else names[::-1]):
                flush.fill_(j);a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                a.record();graphs[name].replay();b.record();b.synchronize();samples[name].append(a.elapsed_time(b)*1000)
        med={k:statistics.median(v) for k,v in samples.items()}
        RESULT['timings'].append(dict(prefix=prefix,group=group,tokens=m,raw_us=samples,median_us=med,
            max_ep8_over_max_ep4=max(med['ep0'],med['ep1'])/max(med['tp0'],med['tp1'])))
    save('shape_complete',current=dict(prefix=prefix,group=group,tokens=m))
def run_group(prefix,group,total,topk,shapes):
    save('loading',current=dict(prefix=prefix,group=group));raw=load_weights(prefix,group,total);E=total//4
    halves=[split(raw,t) for t in range(2)]
    # Lossless partition: concatenate every packed weight/scale back to the original.
    for k in ('w13','s13'):
        h=N//2
        rebuilt=torch.cat([halves[0][k][:,:h],halves[1][k][:,:h],halves[0][k][:,h:],halves[1][k][:,h:]],dim=1)
        assert torch.equal(rebuilt,raw[k]);del rebuilt
    for k in ('w2','s2'):assert torch.equal(torch.cat([h[k] for h in halves],dim=2),raw[k])
    arms={'full':(prepare(raw),1,0,4,group)}
    for t in range(2):
        arms[f'tp{t}']=(prepare(halves[t]),2,t,4,group)
        chunk={k:v[t*(E//2):(t+1)*(E//2)] for k,v in raw.items()}
        arms[f'ep{t}']=(prepare(chunk),1,0,8,group*2+t)
    for m in shapes:check_shape(prefix,group,total,topk,m,arms)
    del arms,halves,raw;gc.collect();torch.cuda.empty_cache()
try:
    assert torch.cuda.get_device_capability()==(12,1)
    RESULT['device']=str(torch.cuda.get_device_properties(0))
    for group in range(4):
        run_group('layers.0',group,384,6,[1,5,6,24,48,128,8192,9215] if group==0 else [6,48])
    for stage in range(3):
        for group in range(4):run_group(f'mtp.{stage}',group,128,3,[5,40] if stage==0 else [5])
    RESULT['status']='PASS';save('complete')
except BaseException as exc:
    RESULT['status']='FAIL';save('failed',error=repr(exc));raise
