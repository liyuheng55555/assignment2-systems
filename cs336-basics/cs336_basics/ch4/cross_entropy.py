import einops
import torch
from jaxtyping import Float, Int
from torch import Tensor


def cross_entropy(
        inputs: Float[Tensor, " batch_size vocab_size"],
        targets: Int[Tensor, " batch_size"],
) -> Float[Tensor, ""]:
    inputs = inputs.float()
    targets = targets.long()
    batch_size = inputs.numel() // inputs.shape[-1]
    max_inputs = torch.max(inputs, dim=1).values
    max_inputs = einops.rearrange(max_inputs, "(batch vocab) -> batch vocab", vocab=1)
    shifted_inputs = inputs - max_inputs
    exp_sum = einops.einsum(torch.exp(shifted_inputs), "batch vocab -> batch")
    correct_logits = shifted_inputs[torch.arange(shifted_inputs.shape[0]), targets] # [0,1...  targets[0],targets[1]...]
    loss = (torch.log(exp_sum) - correct_logits).sum() / batch_size
    return loss
