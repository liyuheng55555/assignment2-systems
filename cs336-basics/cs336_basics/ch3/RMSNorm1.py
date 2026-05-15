import math

import einops
import torch
from torch import nn

class RMSNorm1(nn.Module):
    def __init__(self, d_model: int, weights: torch.Tensor, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.d_model = d_model
        self.g: nn.Parameter = nn.Parameter(weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        x2 = x*x
        x2sum = einops.einsum(x2, "... d_model -> ...")
        rms = (x2sum / self.d_model + self.eps).sqrt()
        rms_1 = 1 / rms
        return self.g * einops.einsum(x, rms_1, "batch sequence d_model, batch sequence -> batch sequence d_model").to(in_dtype)