import math

import einops
import torch
from typing import Optional

from jaxtyping import Float, Int
from torch import Tensor


class RotaryPositionalEmbedding(torch.nn.Module):


    def __init__(self, d_k: int, max_seq_len: int, theta: float = 10000.0, device=None):
        super().__init__()
        self.theta = theta
        self.d = d_k
        self.max_seq_len = max_seq_len
        self.device = device
        self.cal_cos_sin()


    def cal_cos_sin(self):
        i = torch.arange(self.max_seq_len, device=self.device)
        k = torch.arange(self.d // 2, device=self.device)
        extend_i = i[:, None]
        extend_k = k[None, :]
        # [max_seq_len, self.d]
        theta_i_k = extend_i / (self.theta ** ((2*extend_k) / self.d))
        self.cos = theta_i_k.cos()
        self.sin = theta_i_k.sin()


    def forward(
            self,
            x: Float[Tensor, " ... sequence_length d_k"],
            token_positions: Int[Tensor, " ... sequence_length"],
    ) -> Float[Tensor, " ... sequence_length d_k"]:
        # 需要想想怎么优雅化
        COS = self.cos[token_positions].to(x.device) # [... sequence_length self.d // 2]
        SIN = self.sin[token_positions].to(x.device)

        a = x[..., ::2]
        b = x[..., 1::2]

        a_cos = a * COS
        a_sin = a * SIN
        b_cos = b * COS
        b_sin = b * SIN

        g = a_cos - b_sin
        g_1 = a_sin + b_cos

        G = torch.stack([g, g_1], dim=-1) # ... sequence_length, self.d // 2, 2
        result = einops.rearrange(G, "... half_d two -> ... (half_d two)")
        return result


