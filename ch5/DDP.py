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
        
        def all_reduce_hook(x: torch.Tensor):
            dist.all_reduce(x.grad, async_op=False)
            x.grad /= dist.get_world_size()
        
        for parameter in module.parameters():
            dist.broadcast(parameter, src=0)
            if parameter.requires_grad:
                parameter.register_post_accumulate_grad_hook(hook=all_reduce_hook)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        return True

