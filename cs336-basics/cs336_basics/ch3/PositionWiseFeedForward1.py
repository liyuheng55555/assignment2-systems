import math

import einops
import torch
from torch import nn, Tensor
from jaxtyping import Float, Int


class PositionWiseFeedForward1(nn.Module):
    def __init__(self, d_model: int, d_ff: int, w1: Float[Tensor, " d_ff d_model"], w2: Float[Tensor, " d_model d_ff"],
                 w3: Float[Tensor, " d_ff d_model"], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = nn.Parameter(w1)
        self.w2 = nn.Parameter(w2)
        self.w3 = nn.Parameter(w3)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1x = einops.einsum(self.w1, x, "d_ff d_model, ... d_model -> ... d_ff")
        w3x = einops.einsum(self.w3, x, "d_ff d_model, ... d_model -> ... d_ff")
        return einops.einsum(self.w2, self.silu(w1x) * w3x, "d_model d_ff, ... d_ff -> ... d_model")


    def silu(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)
