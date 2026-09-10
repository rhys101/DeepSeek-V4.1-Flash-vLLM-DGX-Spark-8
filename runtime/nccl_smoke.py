"""One GPU per process. Measures payload/time, not nccl-tests bus bandwidth."""
import datetime
import os
import time

import torch
import torch.distributed as dist

torch.cuda.set_device(0)
dist.init_process_group('nccl', timeout=datetime.timedelta(minutes=5), device_id=torch.device('cuda:0'))
rank, world = dist.get_rank(), dist.get_world_size()
print(f'rank={rank}/{world} host={os.uname().nodename}', flush=True)
expected = world * (world + 1) / 2
for n in (1024, 1024 * 1024, 32 * 1024 * 1024):
    x = torch.full((n,), rank + 1, device='cuda', dtype=torch.float32)
    dist.all_reduce(x)
    assert torch.all(x == expected).item(), f'Bad all_reduce on rank {rank}'
    if n == 1024:
        with open('/proc/self/maps') as handle:
            nccl_paths = sorted({line.split()[-1] for line in handle if 'libnccl.so' in line})
        print(f'rank={rank} NCCL version={torch.cuda.nccl.version()} loaded_libraries={nccl_paths}', flush=True)
    x.zero_()
    for _ in range(3):
        dist.all_reduce(x)
    torch.cuda.synchronize()
    dist.barrier()
    t0 = time.perf_counter()
    for _ in range(20):
        dist.all_reduce(x)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) / 20
    if rank == 0:
        print(f'all_reduce bytes={n*4} mean_ms={elapsed*1000:.3f} payload_GBps={n*4/elapsed/1e9:.3f}', flush=True)
    del x

x = torch.full((1024 * 1024,), rank, device='cuda', dtype=torch.float32)
gathered = [torch.empty_like(x) for _ in range(world)]
dist.all_gather(gathered, x)
for i, result in enumerate(gathered):
    assert torch.all(result == i).item(), f'Bad all_gather on rank {rank}, source {i}'
dist.barrier()
print(f'PASS rank={rank}: all_reduce and all_gather; inspect NCCL logs for the actual data transport.', flush=True)
dist.destroy_process_group()
