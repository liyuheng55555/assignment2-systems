import logging
from time import perf_counter

import torch
from einops import einsum

from ch4.FlashAttentionByTriton import FlashAttentionByTriton


def core_loop(Q, K, V, grad_O, op):
    O = FlashAttentionByTriton.apply(Q, K, V, False)
    if op >= 1:
        O.backward(grad_O)


def benchmark(seq_len: int, d: int, dtype, op: int):
    logging.info(f"For {seq_len=} {d=} {dtype=}")
    Q = torch.rand((seq_len, d), dtype=dtype)
    K = torch.rand((seq_len, d), dtype=dtype)
    V = torch.rand((seq_len, d), dtype=dtype)
    grad_O = torch.rand((seq_len, d), dtype=dtype)

    for i in range(5):
        core_loop(Q, K, V, grad_O, op)

    start = perf_counter()
    for i in range(10):
        core_loop(Q, K, V, grad_O, op)
    return (perf_counter() - start) / 10


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    seq_len_list = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
    d_list = [16, 32, 64, 128]
    precisions = torch.bfloat16, torch.float32
    for seq_len in seq_len_list:
        for d in d_list:
            for precision in precisions:

                latency = benchmark(seq_len, d, precision, 0)
