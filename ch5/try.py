import os
from time import perf_counter

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
def distributed_demo(rank, world_size):
    setup(rank, world_size)
    data = torch.rand((1024*1024*1024//torch.float32.itemsize,))
    for i in range(5):
        dist.all_reduce(data, async_op=False)
    # print(f"rank {rank} data (before all-reduce): {data}")
    start = perf_counter()
    dist.all_reduce(data, async_op=False)
    print(f"{rank=} time cost {perf_counter()-start:.4f}s")
    # print(f"rank {rank} data (after all-reduce): {data}")
if __name__ == "__main__":
    world_size = 4
    mp.spawn(fn=distributed_demo, args=(world_size, ), nprocs=world_size, join=True)