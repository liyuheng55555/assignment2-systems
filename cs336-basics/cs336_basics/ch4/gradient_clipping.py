import math
from typing import Iterable

import torch

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    eps = 1e-6
    l2 = 0
    for parameter in parameters:
        if parameter.grad is not None:
            l2 += parameter.grad.pow(2).sum()
    l2_norm = math.sqrt(l2)
    if l2_norm > max_l2_norm:
        scale_factor = max_l2_norm / (l2_norm + eps)
        for parameter in parameters:
            if parameter.grad is not None:
                parameter.grad.data *= scale_factor