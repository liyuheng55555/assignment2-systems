
import torch
from torch import nn

class Embedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.w: nn.Parameter = nn.Parameter(torch.empty(vocab_size, d_model).normal_(mean=0.0, std=0.02))

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.w[token_ids]

