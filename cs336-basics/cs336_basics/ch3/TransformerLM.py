import torch
from jaxtyping import Float, Int

from torch import nn, Tensor

from ch3.Embedding import Embedding
from ch3.Linear import Linear
from ch3.RMSNorm1 import RMSNorm1
from ch3.Softmax import softmax
from ch3.TransformerBlock import TransformerBlock


class TransformerLM(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            context_length: int,
            d_model: int,
            num_layers: int,
            num_heads: int,
            d_ff: int,
            rope_theta: float,
            weights: dict[str, Tensor],
    ):
        super().__init__()

        self.embedding_block = Embedding(vocab_size, d_model)
        self.embedding_block.w = nn.Parameter(weights['token_embeddings.weight'])

        self.transformer_blocks = nn.ModuleList()
        for i in range(num_layers):
            self.transformer_blocks.append(TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta, self.extract_weights(weights, i)))

        self.norm = RMSNorm1(d_model, weights['ln_final.weight'])

        self.linear = Linear(d_model, vocab_size, weights=weights['lm_head.weight'])


    def forward(self, in_indices: Int[Tensor, " batch_size sequence_length"]):
        token_positions = torch.arange(in_indices.shape[-1], device=in_indices.device)

        x = self.embedding_block.forward(in_indices)

        for transformer_block in self.transformer_blocks:
            x = transformer_block.forward(x, token_positions)

        x = self.norm.forward(x)

        x = self.linear.forward(x)

        # x = softmax(x, -1)

        return x


    @staticmethod
    def activations(
            vocab_size: int,
            context_length: int,
            d_model: int,
            num_layers: int,
            num_heads: int,
            d_ff: int,
    ):
        pass




    def extract_weights(self, weights: dict[str, Tensor], id: int):
        keys = [
            'attn.q_proj.weight',
            'attn.k_proj.weight',
            'attn.v_proj.weight',
            'attn.output_proj.weight',

            'ffn.w1.weight',
            'ffn.w2.weight',
            'ffn.w3.weight',

            'ln1.weight',
            'ln2.weight',
        ]

        result: dict[str, Tensor] = {}

        for key in keys:
            key_with_layer_id = f"layers.{id}.{key}"
            result[key] = weights[key_with_layer_id]

        return result