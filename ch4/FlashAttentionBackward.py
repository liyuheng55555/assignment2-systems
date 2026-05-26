import math
from typing import Any
import einops
import torch


def flash_attention_backward(ctx: torch.autograd.function.FunctionCtx, grad_O: torch.Tensor) -> Any:
    L, Q, K, V, O = ctx.saved_tensors
    Kt = einops.rearrange(K, "... N d -> ... d N")
    Vt = einops.rearrange(V, "... N d -> ... d N")
    S = Q @ Kt / math.sqrt(ctx.d)
    if ctx.is_causal:
        mask = torch.tril(torch.ones(S.shape, dtype=torch.bool, device=S.device))
        S = torch.masked_fill(S, ~mask, -1e6)
    P = torch.exp(S - L[..., None])
    Pt = einops.rearrange(P, "... N1 N2 -> ... N2 N1")
    grad_V = Pt @ grad_O
    grad_P = grad_O @ Vt
    D = torch.sum(O * grad_O, dim=-1)
    grad_S = P * (grad_P - D[..., None])
    grad_Q = grad_S @ K / math.sqrt(ctx.d)
    grad_S_t = einops.rearrange(grad_S, "... N1 N2 -> ... N2 N1")
    grad_K = grad_S_t @ Q / math.sqrt(ctx.d)
    return grad_Q, grad_K, grad_V, None