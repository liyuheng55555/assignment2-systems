import math
import torch
from collections.abc import Callable, Iterable
from typing import Optional

from ch3.transformer_accounting import calculate_parameters
from ch4.lr_cosine_schedule import lr_cosine_schedule


class AdamW(torch.optim.Optimizer):
    def __init__(
            self,
            params,
            lr: float,
            weight_decay: float,
            betas: tuple[float, float],
            eps: float,
            device: torch.device,
            cosine_cycle_iters: int = None
    ):
        self._global_step = 0
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "betas": betas,
            "eps": eps,
            "device": device,
            "cosine_cycle_iters": cosine_cycle_iters
        }
        super().__init__(params, defaults)


    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        self._global_step += 1
        current_global_step = self._global_step
        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            b1, b2 = group["betas"]
            eps = group["eps"]
            cosine_cycle_iters = group["cosine_cycle_iters"]

            for p in group["params"]:
                assert isinstance(p, torch.Tensor)
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state["t"] = 1
                    state["m"] = torch.zeros_like(p, dtype=torch.float32)
                    state["v"] = torch.zeros_like(p, dtype=torch.float32)

                t = state.get("t")
                m: torch.Tensor = state.get("m")
                v: torch.Tensor = state.get("v")
                g = p.grad.data

                scheduled_lr = lr
                if cosine_cycle_iters is not None:
                    scheduled_lr = lr_cosine_schedule(current_global_step, lr, lr/10, cosine_cycle_iters // 20, cosine_cycle_iters)

                p.data -= scheduled_lr * weight_decay * p.data

                lr_t = scheduled_lr * math.sqrt(1 - b2 ** t) / (1 - b1 ** t)

                m = b1 * m + (1 - b1) * g
                v = b2 * v + (1 - b2) * (g ** 2)

                p.data -= lr_t * m / (v.sqrt() + eps)

                state["t"] = t + 1
                state["m"] = m
                state["v"] = v


def adamw_accounting(
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
):
    pass