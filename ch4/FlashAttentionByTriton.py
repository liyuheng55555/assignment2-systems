import math
import os
from typing import Any

import einops
import torch

# os.environ['TRITON_INTERPRET'] = '1'

import triton
import triton.language as tl

class FlashAttentionByTriton(torch.autograd.Function):
    @staticmethod
    def forward(
            ctx: torch.autograd.function.FunctionCtx,
            Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, is_causal=False
    ) -> torch.Tensor:
        d = Q.shape[-1]
        N_q = Q.shape[-2]
        N_kv = K.shape[-2]
        b_q = 16
        b_kv = 32
        T_q = (N_q + b_q - 1) // b_q
        T_k = (N_kv + b_kv - 1) // b_kv
        Kt = einops.rearrange(K, "... N_k d -> ... d N_k")

        O = torch.zeros(Q.shape, device=Q.device)
        L = torch.zeros(Q.shape[:-1], device=Q.device)

        FlashAttentionByTriton.forward_kernel[(
            FlashAttentionByTriton.get_batch_count(Q),
            T_q
        )](
            Q, Kt, V,
            O, L,
            Q.stride(-3), Q.stride(-2), Q.stride(-1),
            Kt.stride(-3), Kt.stride(-2), Kt.stride(-1),
            V.stride(-3), V.stride(-2), V.stride(-1),
            O.stride(-3), O.stride(-2), O.stride(-1),
            L.stride(-2), L.stride(-1),
            N_q, N_kv,
            0,
            d,
            b_q,
            b_kv, T_k
        )

        ctx.save_for_backward(L)

        return O

    @staticmethod
    @triton.jit
    def forward_kernel(
            Q_ptr, Kt_ptr, V_ptr,
            O_ptr, L_ptr,
            q_batch_stride, q_N_stride, q_d_stride,
            kt_batch_stride, kt_d_stride, kt_N_stride,
            v_batch_stride, v_N_stride, v_d_stride,
            o_batch_stride, o_N_stride, o_d_stride,
            l_batch_stride, l_N_stride,
            N_q, N_kv,
            scale,
            d: tl.constexpr,
            B_q: tl.constexpr,
            B_kv: tl.constexpr, # 分块尺寸
            T_kv: tl.constexpr, # 多少个分块
    ):

        batch_index = tl.program_id(0)
        q_block_index = tl.program_id(1)

        # Qi = Q[..., range_Q, :]
        q_block_ptr = tl.make_block_ptr(
            base=Q_ptr + batch_index * q_batch_stride,
            shape=(N_q, d),
            strides=(q_N_stride, q_d_stride),
            offsets=(q_block_index * B_q, 0),
            block_shape=(B_q, d),
            order=(1,0),
        )
        q_block = tl.load(q_block_ptr, boundary_check=(0,), padding_option="zero")

        kt_block_ptr = tl.make_block_ptr(
            base=Kt_ptr + batch_index * kt_batch_stride,
            shape=(d, N_kv),
            strides=(kt_d_stride, kt_N_stride),
            offsets=(0, 0),
            block_shape=(d, B_kv),
            order=(0,1) # 这里不确定
        )

        v_block_ptr = tl.make_block_ptr(
            base=V_ptr + batch_index * v_batch_stride,
            shape=(N_kv, d),
            strides=(v_N_stride, v_d_stride),
            offsets=(0, 0),
            block_shape=(B_kv, d),
            order=(1, 0)
        )

        o_block_ptr = tl.make_block_ptr(
            base=O_ptr + batch_index * o_batch_stride,
            shape=(N_q, d),
            strides=(o_N_stride, o_d_stride),
            offsets=(q_block_index * B_q, 0),
            block_shape=(B_q, d),
            order=(1,0),
        )
        o_block = tl.load(o_block_ptr, boundary_check=(0,), padding_option="zero")

        li = tl.zeros((B_q,), dtype=tl.float32)
        mi = tl.full((B_q,), float("-inf"), dtype=tl.float32)

        l_block_ptr = tl.make_block_ptr(
            base=L_ptr + batch_index * l_batch_stride,
            shape=(N_q,),
            strides=(l_N_stride,),
            offsets=(q_block_index * B_q,),
            block_shape=(B_q,),
            order=(0,),
        )

        float_d = float(d)


        for j in range(T_kv):
            kt_block = tl.load(kt_block_ptr, boundary_check=(1,), padding_option="zero")
            v_block = tl.load(v_block_ptr, boundary_check=(0,), padding_option="zero")
            part_S = tl.dot(q_block, kt_block) / tl.sqrt(float_d)
            mi_new = tl.maximum(mi, tl.max(part_S, axis=-1))
            P = tl.exp(part_S - mi_new[:, None])
            exp_correction = tl.exp(mi - mi_new)
            li = exp_correction * li + tl.sum(P, axis=-1)
            o_block = exp_correction[:, None] * o_block + tl.dot(P, v_block)

            mi = mi_new
            kt_block_ptr = kt_block_ptr.advance((0, B_kv))
            v_block_ptr = v_block_ptr.advance((B_kv, 0))

        o_block = (1/li)[:, None] * o_block
        l_block = mi + tl.log(li)
        tl.store(o_block_ptr, o_block, boundary_check=(0,))
        tl.store(l_block_ptr, l_block, boundary_check=(0,))

    @staticmethod
    def get_batch_count(X: torch.Tensor) -> int:
        shapes = X.shape[:-2]
        result = 1
        for shape in shapes:
            result *= shape
        return result


    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        pass