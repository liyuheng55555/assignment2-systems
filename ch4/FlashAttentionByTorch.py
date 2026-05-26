import math
from typing import Any

import einops
import torch


class FlashAttentionByTorch(torch.autograd.Function):
    @staticmethod
    def forward(
            ctx: torch.autograd.function.FunctionCtx,
            Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, is_causal=False
    ) -> torch.Tensor:
        d = torch.full((1,), float(Q.shape[-1]))
        N_q = Q.shape[-2]
        N_kv = K.shape[-2]
        b_q = 15
        b_kv = 33
        T_q = (N_q + b_q - 1) // b_q
        T_k = (N_kv + b_kv - 1) // b_kv
        Kt = einops.rearrange(K, "... N_k d -> ... d N_k")
        O = torch.empty(Q.shape)
        L = torch.empty(Q.shape[:-1])
        for i in range(T_q):
            range_Q = slice(i*b_q, min((i+1)*b_q, N_q))
            Qi = Q[..., range_Q, :]
            Oi = torch.zeros(Qi.shape)
            li = torch.zeros(Qi.shape[:-1])
            mi = torch.full(Qi.shape[:-1], float("-inf"))
            for j in range(T_k):
                range_kv = slice(j*b_kv, min((j+1)*b_kv, N_kv))
                Ktj = Kt[..., range_kv]
                Vj = V[..., range_kv, :]
                part_S = (Qi @ Ktj) / math.sqrt(d)
                mi_new =  torch.maximum(mi, part_S.max(dim=-1).values)
                P = torch.exp(part_S - mi_new[..., None])
                li = torch.exp(mi - mi_new) * li + P.sum(dim=-1)
                Oi = torch.diag_embed(torch.exp(mi - mi_new)) @ Oi + P @ Vj
                mi = mi_new
            Oi = torch.diag_embed(li).inverse() @ Oi
            Li = mi + li.log()
            O[..., range_Q, :] = Oi
            L[..., range_Q] = Li
        ctx.save_for_backward(L)
        return O

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        pass

