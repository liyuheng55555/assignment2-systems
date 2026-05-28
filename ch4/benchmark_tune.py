from __future__ import annotations

import argparse
import csv
import logging
import math
from datetime import datetime
from pathlib import Path
from time import perf_counter

import einops
import torch
from triton.runtime.errors import OutOfResources

from ch4.FlashAttentionByTriton import FlashAttentionByTriton


def parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _is_power_of_two(x: int) -> bool:
    return x > 0 and (x & (x - 1)) == 0


def _prev_power_of_two(x: int) -> int:
    if x < 1:
        return 1
    return 1 << (x.bit_length() - 1)


def _default_block_candidates(d: int, dtype: torch.dtype, shared_memory_bytes: int) -> list[int]:
    itemsize = torch.tensor([], dtype=dtype).element_size()
    m_elements = shared_memory_bytes // itemsize
    center = _prev_power_of_two(max(1, m_elements // (4 * d)))
    exp = int(math.log2(center))
    cands = sorted({1 << max(0, exp - 1), 1 << exp, 1 << (exp + 1)})
    return cands


def triton_attention_with_blocks(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b_q: int,
    b_kv: int,
    is_causal: bool = False,
) -> torch.Tensor:
    d = q.shape[-1]
    n_q = q.shape[-2]
    n_kv = k.shape[-2]
    t_q = (n_q + b_q - 1) // b_q
    t_kv = (n_kv + b_kv - 1) // b_kv

    kt = einops.rearrange(k, "... n_k d -> ... d n_k")
    o = torch.zeros(q.shape, device=q.device)
    l = torch.zeros(q.shape[:-1], device=q.device)

    FlashAttentionByTriton.forward_kernel[(FlashAttentionByTriton.get_batch_count(q), t_q)](
        q,
        kt,
        v,
        o,
        l,
        q.stride(-3),
        q.stride(-2),
        q.stride(-1),
        kt.stride(-3),
        kt.stride(-2),
        kt.stride(-1),
        v.stride(-3),
        v.stride(-2),
        v.stride(-1),
        o.stride(-3),
        o.stride(-2),
        o.stride(-1),
        l.stride(-2),
        l.stride(-1),
        n_q,
        n_kv,
        0,
        d,
        b_q,
        b_kv,
        t_kv,
        is_causal,
    )

    return o


def benchmark_one(
    seq_len: int,
    d: int,
    dtype: torch.dtype,
    b_q: int,
    b_kv: int,
    warmup: int,
    repeat: int,
    is_causal: bool,
) -> tuple[float, float, str]:
    device = torch.device("cuda")

    q = torch.rand((1, seq_len, d), dtype=dtype, device=device)
    k = torch.rand((1, seq_len, d), dtype=dtype, device=device)
    v = torch.rand((1, seq_len, d), dtype=dtype, device=device)

    try:
        torch.cuda.empty_cache()
        for _ in range(warmup):
            triton_attention_with_blocks(q, k, v, b_q=b_q, b_kv=b_kv, is_causal=is_causal)

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        start = perf_counter()
        for _ in range(repeat):
            triton_attention_with_blocks(q, k, v, b_q=b_q, b_kv=b_kv, is_causal=is_causal)
            torch.cuda.synchronize()
        latency_ms = (perf_counter() - start) / repeat * 1000.0
        peak_mem_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        return latency_ms, peak_mem_mb, "ok"
    except OutOfResources as e:
        msg = str(e)
        return -1.0, -1.0, f"out_of_resources: {msg[:200]}"
    except RuntimeError as e:
        msg = str(e)
        if "out of memory" in msg.lower():
            return -1.0, -1.0, "oom"
        return -1.0, -1.0, f"runtime_error: {msg[:200]}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune Triton FlashAttention block sizes (b_q, b_kv)")
    parser.add_argument("--seq-lens", type=str, default="128,256,512,1024,2048,4096,8192,16384")
    parser.add_argument("--dims", type=str, default="16,32,64,128")
    parser.add_argument("--b-qs", type=str, default=None, help="Override b_q candidates, e.g. 16,32,64")
    parser.add_argument("--b-kvs", type=str, default=None, help="Override b_kv candidates, e.g. 16,32,64")
    parser.add_argument("--shared-memory", type=int, default=100 * 1024, help="Bytes, used for default candidate generation")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--topk", type=int, default=0, help="Print top-k fastest configs per (seq_len, d); 0 disables")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    seq_lens = parse_int_list(args.seq_lens)
    dims = parse_int_list(args.dims)

    user_b_qs = parse_int_list(args.b_qs) if args.b_qs else None
    user_b_kvs = parse_int_list(args.b_kvs) if args.b_kvs else None

    if user_b_qs is not None and any(not _is_power_of_two(x) for x in user_b_qs):
        raise ValueError("All --b-qs values must be powers of two.")
    if user_b_kvs is not None and any(not _is_power_of_two(x) for x in user_b_kvs):
        raise ValueError("All --b-kvs values must be powers of two.")

    out_dir = Path("ch4") / "benchmark_result"
    out_dir.mkdir(parents=True, exist_ok=True)
    output_csv = args.out or (out_dir / f"tune_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logging.info("Writing results to %s", output_csv)

    all_rows: list[dict] = []
    for seq_len in seq_lens:
        for d in dims:
            default_cands = _default_block_candidates(d=d, dtype=dtype, shared_memory_bytes=args.shared_memory)
            b_qs = user_b_qs if user_b_qs is not None else default_cands
            b_kvs = user_b_kvs if user_b_kvs is not None else default_cands

            logging.info(
                "Testing seq_len=%d d=%d | default_candidates=%s | b_qs=%s b_kvs=%s",
                seq_len,
                d,
                default_cands,
                b_qs,
                b_kvs,
            )
            group_rows: list[dict] = []
            for b_q in b_qs:
                for b_kv in b_kvs:
                    latency, peak_mem, status = benchmark_one(
                        seq_len=seq_len,
                        d=d,
                        dtype=dtype,
                        b_q=b_q,
                        b_kv=b_kv,
                        warmup=args.warmup,
                        repeat=args.repeat,
                        is_causal=args.causal,
                    )
                    row = {
                        "dtype": str(dtype),
                        "seq_len": seq_len,
                        "d": d,
                        "b_q": b_q,
                        "b_kv": b_kv,
                        "causal": int(args.causal),
                        "latency_ms": latency,
                        "peak_memory_mb": peak_mem,
                        "status": status,
                        "is_best_latency": 0,
                    }
                    all_rows.append(row)
                    group_rows.append(row)

            ok_rows = [r for r in group_rows if r["status"] == "ok" and r["latency_ms"] > 0]
            if not ok_rows:
                logging.warning("No successful config for seq_len=%d d=%d", seq_len, d)
                continue

            best = min(ok_rows, key=lambda r: r["latency_ms"])
            best["is_best_latency"] = 1

            if args.topk > 0:
                k = min(args.topk, len(ok_rows))
                top_rows = sorted(ok_rows, key=lambda r: r["latency_ms"])[:k]
                logging.info("Top-%d configs for seq_len=%d d=%d:", k, seq_len, d)
                for i, r in enumerate(top_rows, start=1):
                    logging.info(
                        "  #%d b_q=%d b_kv=%d latency=%.6f ms peak_mem=%.3f MB",
                        i,
                        r["b_q"],
                        r["b_kv"],
                        r["latency_ms"],
                        r["peak_memory_mb"],
                    )

    fieldnames = [
        "dtype",
        "seq_len",
        "d",
        "b_q",
        "b_kv",
        "causal",
        "latency_ms",
        "peak_memory_mb",
        "status",
        "is_best_latency",
    ]

    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    logging.info("Done. %d rows written.", len(all_rows))


if __name__ == "__main__":
    main()
