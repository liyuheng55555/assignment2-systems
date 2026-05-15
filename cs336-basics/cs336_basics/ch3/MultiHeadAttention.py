import logging
import math

import einops
import torch
from jaxtyping import Float, Int
from torch import Tensor, nn

from ch3.RotaryPositionalEmbedding import RotaryPositionalEmbedding
from ch3.Softmax import softmax


class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            q_proj_weight: Float[Tensor, " d_model d_model"],
            k_proj_weight: Float[Tensor, " d_model d_model"],
            v_proj_weight: Float[Tensor, " d_model d_model"],
            o_proj_weight: Float[Tensor, " d_model d_model"],
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.W_q : nn.Parameter = nn.Parameter(q_proj_weight)
        self.W_k : nn.Parameter = nn.Parameter(k_proj_weight)
        self.W_v : nn.Parameter = nn.Parameter(v_proj_weight)
        self.W_o : nn.Parameter = nn.Parameter(o_proj_weight)


    def forward(self, in_features: Float[Tensor, " ... sequence_length d_model"]
                ) -> Float[Tensor, " ... sequence_length d_model"]:
        multi_q = einops.einsum(in_features, self.W_q, "... seq_len d_model, d_multi_q d_model -> ... seq_len d_multi_q")
        multi_k = einops.einsum(in_features, self.W_k, "... seq_len d_model, d_multi_k d_model -> ... seq_len d_multi_k")
        multi_v = einops.einsum(in_features, self.W_v, "... seq_len d_model, d_multi_v d_model -> ... seq_len d_multi_v")

        multi_q = einops.rearrange(multi_q, "... seq_len (heads d_q) -> ... heads seq_len d_q", heads=self.num_heads)
        multi_k = einops.rearrange(multi_k, "... seq_len (heads d_k) -> ... heads seq_len d_k", heads=self.num_heads)
        multi_v = einops.rearrange(multi_v, "... seq_len (heads d_v) -> ... heads seq_len d_v", heads=self.num_heads)

        seq_len : int = in_features.shape[-2]

        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))

        attention = scaled_dot_product_attention(multi_q, multi_k, multi_v, mask)

        concat = einops.rearrange(attention, "... heads seq_len d_v -> ... seq_len (heads d_v)") # heads * d_v == d_model

        result = einops.einsum(concat, self.W_o, "... seq_len d_model, d_out d_model-> ... seq_len d_out")

        return result


class MultiHeadAttentionWithRope(nn.Module):
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            max_seq_len: int,
            theta: float,
            q_proj_weight: Float[Tensor, " d_model d_model"],
            k_proj_weight: Float[Tensor, " d_model d_model"],
            v_proj_weight: Float[Tensor, " d_model d_model"],
            o_proj_weight: Float[Tensor, " d_model d_model"],
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        # assert q_proj_weight.shape[0] % num_heads == 0
        self.d_k = q_proj_weight.shape[0] // num_heads
        self.W_q : nn.Parameter = nn.Parameter(q_proj_weight)
        self.W_k : nn.Parameter = nn.Parameter(k_proj_weight)
        self.W_v : nn.Parameter = nn.Parameter(v_proj_weight)
        self.W_o : nn.Parameter = nn.Parameter(o_proj_weight)
        self.rope = RotaryPositionalEmbedding(self.d_k, max_seq_len, theta, device=self.W_k.device)


    def forward(
            self,
            in_features: Float[Tensor, " ... sequence_length d_model"],
            token_positions: Int[Tensor, " ... sequence_length"] | None = None,
    ) -> Float[Tensor, " ... sequence_length d_model"]:
        multi_q = einops.einsum(in_features, self.W_q, "... seq_len d_model, d_multi_q d_model -> ... seq_len d_multi_q")
        multi_k = einops.einsum(in_features, self.W_k, "... seq_len d_model, d_multi_k d_model -> ... seq_len d_multi_k")
        multi_v = einops.einsum(in_features, self.W_v, "... seq_len d_model, d_multi_v d_model -> ... seq_len d_multi_v")

        multi_q = einops.rearrange(multi_q, "... seq_len (heads d_q) -> ... heads seq_len d_q", heads=self.num_heads)
        multi_k = einops.rearrange(multi_k, "... seq_len (heads d_k) -> ... heads seq_len d_k", heads=self.num_heads)
        multi_v = einops.rearrange(multi_v, "... seq_len (heads d_v) -> ... heads seq_len d_v", heads=self.num_heads)

        rope_q = self.rope.forward(multi_q, token_positions)
        rope_k = self.rope.forward(multi_k, token_positions)

        seq_len : int = in_features.shape[-2]

        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool)).to(in_features.device)

        attention = scaled_dot_product_attention(rope_q, rope_k, multi_v, mask)

        concat = einops.rearrange(attention, "... heads seq_len d_v -> ... seq_len (heads d_v)") # heads * d_v == d_model

        result = einops.einsum(concat, self.W_o, "... seq_len d_model, d_out d_model-> ... seq_len d_out")

        return result

# Q "heads  seq_len  d_q(d_model/num_heads)"
def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor = None):
    d_k: int = Q.shape[-1]
    qkt = einops.einsum(Q, K, "... seq_len_q d_k, ... seq_len_k d_k -> ... seq_len_q seq_len_k") # seq_len_q == seq_len_k == seq_len
    scaled_qkt = qkt / math.sqrt(d_k)
    if mask is not None:
        scaled_qkt = scaled_qkt.masked_fill(~mask, float("-inf"))
    softmax_qk = softmax(scaled_qkt, scaled_qkt.dim() - 1).to(dtype=V.dtype)
    result = einops.einsum(softmax_qk, V, "... seq_len_q seq_len_k, ... seq_len_k d_v -> ... seq_len_q d_v")
    return result
