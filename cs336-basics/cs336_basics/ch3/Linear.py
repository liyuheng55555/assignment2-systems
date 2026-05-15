import math

import einops
import torch
from torch import nn
from einops import rearrange, einsum


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, weights: torch.Tensor = None, device=None, dtype=None):
        super().__init__()
        std = math.sqrt(2/(in_features + out_features))
        self.w = nn.Parameter(weights)
        torch.nn.init.trunc_normal_(self.w, mean=0, std=std, a=-3*std, b=3*std)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einops.einsum(self.w, x, "d_out d_in, ... d_in -> ... d_out")
        # return torch.nn.functional.linear(x, self.w)