import csv
import logging
from datetime import datetime
from pathlib import Path
from time import perf_counter

import torch
from einops import einsum

from ch4.FlashAttentionByTriton import FlashAttentionByTriton


def core_loop(Q, K, V, grad_O, op):
    O = FlashAttentionByTriton.apply(Q, K, V, False)
    if op >= 1:
        O.backward(grad_O)


def benchmark(seq_len: int, d: int, dtype, op: int):
    def _attention_and_lse(q, k, v, grad_o, op):
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

    def core_loop(Q, K, V, grad_O, op):
        O = FlashAttentionByTriton.apply(Q, K, V, False)
        if op >= 1:
            O.backward(grad_O)

    f = _attention_and_lse

    logging.info(f"For {seq_len=} {d=} {dtype=}")
    device = torch.device("cuda")
    try:
        Q = torch.rand((1, seq_len, d), dtype=dtype, device=device)
        K = torch.rand((1, seq_len, d), dtype=dtype, device=device)
        V = torch.rand((1, seq_len, d), dtype=dtype, device=device)
        grad_O = torch.rand((1, seq_len, d), dtype=dtype, device=device)

        torch.cuda.empty_cache()

        for i in range(5):
            f(Q, K, V, grad_O, op)

        torch.cuda.synchronize()
        torch.cuda.reset_max_memory_allocated()
        start = perf_counter()
        for i in range(10):
            f(Q, K, V, grad_O, op)
            torch.cuda.synchronize()
        return (perf_counter() - start) / 10 * 1000, torch.cuda.max_memory_allocated()
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logging.warning(f"OOM for {seq_len=} {d=} {dtype=}, set result to -1")
            return -1
        raise


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    seq_len_list = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
    d_list = [16, 32, 64, 128]
    precisions = torch.bfloat16, torch.float32

    output_dir = Path(__file__).with_name("benchmark_result")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv = output_dir / f"{timestamp}.csv"

    with output_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seq_len", "d", "dtype", "op", "latency"])
        for precision in precisions:
            for seq_len in seq_len_list:
                for d in d_list:
                    latency = benchmark(seq_len, d, precision, 0)
                    writer.writerow([seq_len, d, str(precision), 0, latency])
