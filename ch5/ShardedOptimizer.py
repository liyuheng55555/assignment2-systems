from typing import Type, Any, Callable

from torch.optim import Optimizer

import torch.distributed as dist

class ShardedOptimizer(Optimizer):

    def __init__(self, params, optimizer_cls: Type[Optimizer], **kwargs: Any):
        sharded_params = []
        all_params = []
        for i, param in enumerate(params):
            all_params.append(param)
            if i % dist.get_world_size() == dist.get_rank():
                sharded_params.append(param)

        self.optimizer = optimizer_cls(sharded_params, **kwargs)
        self.all_params = all_params
        super().__init__(sharded_params, kwargs)


    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        self.optimizer.step(closure)

        for i, param in enumerate(self.all_params):
            dist.broadcast(param, src=i % dist.get_world_size())




