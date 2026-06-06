import os
from time import perf_counter

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def distributed_demo(rank, world_size):
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")
    data = torch.rand((5*1024*1024*1024//torch.float32.itemsize,), device=device)
    for i in range(5):
        dist.all_reduce(data, async_op=False)
    # print(f"rank {rank} data (before all-reduce): {data}")
    torch.cuda.synchronize()
    start = perf_counter()
    for i in range(10):
        dist.all_reduce(data, async_op=False)
        torch.cuda.synchronize()
    print(f"{rank=} time cost for one round {(perf_counter()-start)/10:.6f}s")
    # print(f"rank {rank} data (after all-reduce): {data}")
    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 2
    mp.spawn(fn=distributed_demo, args=(world_size, ), nprocs=world_size, join=True)