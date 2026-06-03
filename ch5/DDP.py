import threading
from time import sleep

import torch.nn
import torch.distributed as dist


class NaiveDDP(torch.nn.Module):

    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module
        for parameter in module.parameters():
            dist.broadcast(parameter, src=0)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class OverlapDDP(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module
        self.handles = []
        self.lock = threading.Lock()
        self.grad_count = 0
        
        def all_reduce_hook(x: torch.Tensor):
            handle = dist.all_reduce(x.grad, async_op=True)
            with self.lock:
                self.handles.append(handle)
            # x.grad /= dist.get_world_size()
        
        for parameter in self.module.parameters():
            dist.broadcast(parameter, src=0)
            if parameter.requires_grad:
                parameter.register_post_accumulate_grad_hook(hook=all_reduce_hook)
                self.grad_count += 1


    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


    def finish_gradient_synchronization(self):
        while len(self.handles) < self.grad_count:
            sleep(0.1)
        for handle in self.handles:
            handle.wait()
        with self.lock:
            self.handles.clear()
        for parameter in self.module.parameters():
            if parameter.grad is not None:
                parameter.grad /= dist.get_world_size()
        return


