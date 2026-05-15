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

        # [max_seq_len, self.d]
        self.cos = torch.zeros(max_seq_len, self.d // 2).to(device)
        self.sin = torch.zeros(max_seq_len, self.d // 2).to(device)
        self.cal_cos_sin()


    def cal_cos_sin(self):
        for i in range(self.max_seq_len):
            for k in range(0, self.d // 2):
                theta_i_k = i / (self.theta ** ((2*k) / self.d))
                self.cos[i][k] = math.cos(theta_i_k)
                self.sin[i][k] = math.sin(theta_i_k)


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

        # x_flat = x.flatten()
        # result = torch.zeros(x.numel())
        # base_index = 0
        # while base_index < x.numel() / x.shape[-1]:
        #     position = token_positions.flatten()[base_index % token_positions.numel()]
        #     global_index = base_index * self.d
        #     for g in range(0, self.d, 2):
        #         k = int(g/2) + 1
        #         a = x_flat[global_index + g]
        #         b = x_flat[global_index + g + 1]
        #         result[global_index + g] = a * self.cos[position][k] - b * self.sin[position][k]
        #         result[global_index + g + 1] = a * self.sin[position][k] + b * self.cos[position][k]
        #     base_index += 1
        #
        # result_reshape = result.reshape(x.shape)
        # return result_reshape


