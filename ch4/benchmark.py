import csv
import logging
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Callable

import torch
from einops import einsum

from ch4.FlashAttentionByTriton import FlashAttentionByTriton


def triton_attention(Q, K, V, grad_O, op):
    O = FlashAttentionByTriton.apply(Q, K, V, False)
    if op >= 1:
        O.backward(grad_O)


def torch_attention(q, k, v, grad_O, op):
    n_queries = q.shape[-2]
    n_keys = k.shape[-2]
    d = q.shape[-1]
    scale = 1 / (d**0.5)
    S = einsum(q, k, "... q d, ... k d -> ... q k") * scale
    S = torch.where(
        torch.arange(n_queries, device=S.device)[None, :, None] >= torch.arange(n_keys, device=S.device)[None, None, :],
        S,
        -1e6,
        )
    P = torch.softmax(S, dim=-1)
    o = einsum(P, v, "... q k, ... k d -> ... q d")
    # L = torch.logsumexp(S, dim=-1)
    if op >= 1:
        o.backward(grad_O)
    return o


def benchmark(attention: Callable, seq_len: int, d: int, dtype, op: int):
    logging.info(f"For {attention.__name__} {seq_len=} {d=} {dtype=}")
    device = torch.device("cuda")
    try:
        Q = torch.rand((1, seq_len, d), dtype=dtype, device=device)
        K = torch.rand((1, seq_len, d), dtype=dtype, device=device)
        V = torch.rand((1, seq_len, d), dtype=dtype, device=device)
        grad_O = torch.rand((1, seq_len, d), dtype=dtype, device=device)

        torch.cuda.empty_cache()

        for i in range(5):
            attention(Q, K, V, grad_O, op)

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = perf_counter()
        for i in range(10):
            attention(Q, K, V, grad_O, op)
            torch.cuda.synchronize()
        return (perf_counter() - start) / 10 * 1000, torch.cuda.max_memory_allocated()/1024/1024
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logging.warning(f"OOM for {seq_len=} {d=} {dtype=}, set result to -1")
            return -1, -1
        raise

#### triton有类型问题，设定类型会大幅降速。还有内存统计问题。
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    attentions = torch_attention, triton_attention
    precisions = [
        torch.bfloat16,
        # torch.float32
    ]
    seq_len_list = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
    d_list = [16, 32, 64, 128]

    output_dir = Path(__file__).with_name("benchmark_result")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv = output_dir / f"{timestamp}.csv"

    with output_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["base", "dtype", "seq_len", "d", "op", "latency", "peak_memory"])
        for attention in attentions:
            for precision in precisions:
                for seq_len in seq_len_list:
                    for d in d_list:
                        latency, memory = benchmark(attention, seq_len, d, precision, 0)
                        writer.writerow([attention.__name__, str(precision), seq_len, d, 0, latency, memory])
