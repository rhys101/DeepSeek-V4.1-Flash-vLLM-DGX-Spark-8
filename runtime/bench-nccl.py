"""Eight-rank correctness, graph latency, and host-memory communication probe."""
import argparse
import datetime
import json
import os
import pathlib
import statistics
import torch
import torch.distributed as dist

parser=argparse.ArgumentParser()
parser.add_argument('--out',type=pathlib.Path,required=True)
args=parser.parse_args()
args.out.mkdir(parents=True,exist_ok=True)
torch.cuda.set_device(0)
dist.init_process_group('nccl',timeout=datetime.timedelta(minutes=4),device_id=torch.device('cuda:0'))
rank=dist.get_rank();world=dist.get_world_size()
assert world==8

def memory():
    return {k:int(v.strip().split()[0])*1024 for k,v in (l.split(':',1) for l in pathlib.Path('/proc/meminfo').read_text().splitlines()) if k in ('MemAvailable','Shmem')}

rows=[]
# Match the two independent communicator groups observed in vLLM's startup.
groups=[dist.group.WORLD,dist.new_group(list(range(world)),backend='nccl')]
for gi,group in enumerate(groups):
    for count in [5120,30720,245760,4194304,16777216]:
        x=torch.full((count,),rank+1,device='cuda',dtype=torch.float32)
        dist.all_reduce(x,group=group)
        assert bool((x==36).all()),(rank,gi,count)
        x.zero_()
        for _ in range(5):dist.all_reduce(x,group=group)
        torch.cuda.synchronize();dist.barrier(group=group)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(20):dist.all_reduce(x,group=group)
        durations=[]
        for _ in range(7):
            a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
            a.record();graph.replay();b.record();b.synchronize()
            durations.append(a.elapsed_time(b)*1000/20)
        rows.append(dict(group=gi,bytes=count*4,median_us=statistics.median(durations),samples_us=durations))
        del graph,x
    x=torch.full((4096,),rank,device='cuda',dtype=torch.float32)
    output=torch.empty((world*4096,),device='cuda',dtype=torch.float32)
    dist.all_gather_into_tensor(output,x,group=group)
    assert all(bool((output[i*4096:(i+1)*4096]==i).all()) for i in range(world))
dist.barrier()
result=dict(status='PASS',rank=rank,nccl=torch.cuda.nccl.version(),memory=memory(),env={k:v for k,v in os.environ.items() if k.startswith('NCCL_')},rows=rows)
(args.out/f'rank-{rank}.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
dist.destroy_process_group(groups[1]);dist.destroy_process_group()
