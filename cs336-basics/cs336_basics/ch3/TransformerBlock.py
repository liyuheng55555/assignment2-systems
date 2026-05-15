import torch
from jaxtyping import Float

from torch import nn, Tensor

from ch3.MultiHeadAttention import MultiHeadAttention, MultiHeadAttentionWithRope
from ch3.PositionWiseFeedForward1 import PositionWiseFeedForward1
from ch3.RMSNorm1 import RMSNorm1


class TransformerBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            d_ff: int,
            max_seq_len: int,
            theta: float,
            weights: dict[str, Tensor],
    ):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len

        self.norm1 = RMSNorm1(self.d_model, weights['ln1.weight'])

        self.multihead_attention = MultiHeadAttentionWithRope(
            d_model,
            num_heads,
            max_seq_len,
            theta,
            weights['attn.q_proj.weight'     ],
            weights['attn.k_proj.weight'     ],
            weights['attn.v_proj.weight'     ],
            weights['attn.output_proj.weight']
        )

        self.norm2 = RMSNorm1(self.d_model, weights['ln2.weight'])

        self.position_wise_feed_forward = PositionWiseFeedForward1(
            d_model,
            d_ff,
            weights['ffn.w1.weight'],
            weights['ffn.w2.weight'],
            weights['ffn.w3.weight']
        )


    def forward(
            self,
            in_features: Float[Tensor, " batch sequence_length d_model"],
            token_positions: Float[Tensor, " batch sequence_length"]
    ) -> Float[Tensor, " batch sequence_length d_model"]:
        x = self.norm1.forward(in_features)
        x = self.multihead_attention.forward(x, token_positions)
        step1 = x + in_features
        x = self.norm2.forward(step1)
        x = self.position_wise_feed_forward(x)
        result = x + step1
        return result
