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