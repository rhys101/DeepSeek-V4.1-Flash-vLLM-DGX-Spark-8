"""Idle-cluster TP8 collective correctness with the candidate image's NCCL."""
import datetime
import json
import time
import torch
import torch.distributed as dist

torch.set_num_threads(2)
torch.cuda.set_device(0)
dist.init_process_group('nccl', timeout=datetime.timedelta(seconds=180))
rank=dist.get_rank();assert dist.get_world_size()==8
results=[]
for count in (16384,524288,8388608):
    x=torch.full((count,),float(rank+1),device='cuda',dtype=torch.bfloat16)
    dist.all_reduce(x);torch.cuda.synchronize();assert bool((x==36).all())
    elapsed=[]
    for _ in range(5):
        x.fill_(rank+1);dist.barrier();torch.cuda.synchronize();start=time.perf_counter()
        dist.all_reduce(x);torch.cuda.synchronize();elapsed.append(time.perf_counter()-start)
        assert bool((x==36).all())
    results.append(dict(bytes=count*2,mean_seconds=sum(elapsed)/len(elapsed)))
dist.barrier()
print(json.dumps(dict(status='PASS',rank=rank,world_size=8,nccl=torch.cuda.nccl.version(),all_reduce=results)),flush=True)
dist.destroy_process_group()
