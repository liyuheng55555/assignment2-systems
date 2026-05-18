import logging
import time
from pathlib import Path
import csv

import torch

from cs336_basics.ch3.MultiHeadAttention import MultiHeadAttentionWithRope


def weights_init(
        vocab_size: int,
        d_model: int,
        num_layers: int,
        d_ff: int,
        device: torch.device,
        data_type: torch.dtype,
) -> dict:
    weights = {
        'token_embeddings.weight': torch.empty(vocab_size, d_model, device=device, dtype=data_type).normal_(mean=0.0, std=0.02),
        'ln_final.weight': torch.ones(d_model, device=device, dtype=data_type),
        'lm_head.weight': torch.empty(vocab_size, d_model, device=device, dtype=data_type).normal_(mean=0, std=0.02),
    }
    for i in range(num_layers):
        layer_weights = {
            f'layers.{i}.attn.q_proj.weight': torch.empty(d_model, d_model, device=device, dtype=data_type).normal_(mean=0.0, std=0.02),
            f'layers.{i}.attn.k_proj.weight': torch.empty(d_model, d_model, device=device, dtype=data_type).normal_(mean=0.0, std=0.02),
            f'layers.{i}.attn.v_proj.weight': torch.empty(d_model, d_model, device=device, dtype=data_type).normal_(mean=0.0, std=0.02),
            f'layers.{i}.attn.output_proj.weight': torch.empty(d_model, d_model, device=device, dtype=data_type).normal_(mean=0.0, std=0.02),
            f'layers.{i}.ffn.w1.weight': torch.empty(d_ff, d_model, device=device, dtype=data_type).normal_(mean=0.0, std=0.02),
            f'layers.{i}.ffn.w2.weight': torch.empty(d_model, d_ff, device=device, dtype=data_type).normal_(mean=0.0, std=0.02),
            f'layers.{i}.ffn.w3.weight': torch.empty(d_ff, d_model, device=device, dtype=data_type).normal_(mean=0.0, std=0.02),
            f'layers.{i}.ln1.weight': torch.ones(d_model, device=device, dtype=data_type),
            f'layers.{i}.ln2.weight': torch.ones(d_model, device=device, dtype=data_type),
        }
        weights |= layer_weights
    return weights

def attention_benchmark(
        d_model: int,
        seq_len: int,
        output_dir: Path,
        device: torch.Device = torch.device("cuda"),
        dtype: torch.dtype = torch.bfloat16
) -> (float, float):
    logging.info(f"for d_model={d_model}, seq_len={seq_len}")
    logging.info(f"init model")
    model = MultiHeadAttentionWithRope(
        d_model,
        1,
        seq_len,
        10000,
        torch.empty(d_model, d_model, device=device, dtype=dtype).normal_(mean=0.0, std=0.02),
        torch.empty(d_model, d_model, device=device, dtype=dtype).normal_(mean=0.0, std=0.02),
        torch.empty(d_model, d_model, device=device, dtype=dtype).normal_(mean=0.0, std=0.02),
        torch.empty(d_model, d_model, device=device, dtype=dtype).normal_(mean=0.0, std=0.02),
    )

    x = torch.rand(seq_len, d_model)
    token_positions = torch.arange(0, seq_len)
    for i in range(5):
        x = model.forward(x, token_positions)
        x.backward()

    torch.cuda.memory._record_memory_history(max_entries=1000000)

    logging.info("forward")
    forward_start = time.perf_counter()
    for i in range(100):
        x = model.forward(x, token_positions)
        torch.cuda.synchronize()
        if i % 10 == 0:
            logging.info(f"forward {i}")
    forward_time = time.perf_counter() - forward_start

    snapshot_path = output_dir / f"memory_snapshot_d{d_model}_s{seq_len}.pickle"
    torch.cuda.memory._dump_snapshot(str(snapshot_path))
    torch.cuda.memory._record_memory_history(enabled=None)

    logging.info("backward")
    backward_start = time.perf_counter()
    for i in range(100):
        x.backward()
        torch.cuda.synchronize()
        if i % 10 == 0:
            logging.info(f"backward {i}")
    backward_time = time.perf_counter() - backward_start

    return forward_time, backward_time


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    d_model_list = [16, 32, 64, 128]
    seq_len_list = [256, 1024, 4096, 8192, 16384]

    test_dir = Path("attention_benchmark_results") / time.strftime("%Y%m%d_%H%M%S")
    test_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Saving benchmark outputs to: {test_dir.resolve()}")

    forward_results = {seq_len: {} for seq_len in seq_len_list}
    backward_results = {seq_len: {} for seq_len in seq_len_list}

    for d_model in d_model_list:
        for seq_len in seq_len_list:
            forward_time, backward_time = attention_benchmark(d_model, seq_len, output_dir=test_dir)
            forward_results[seq_len][d_model] = forward_time
            backward_results[seq_len][d_model] = backward_time

    forward_csv_path = test_dir / "forward.csv"
    backward_csv_path = test_dir / "backward.csv"

    with forward_csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seq_len"] + d_model_list)
        for seq_len in seq_len_list:
            writer.writerow([seq_len] + [forward_results[seq_len][d_model] for d_model in d_model_list])

    with backward_csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seq_len"] + d_model_list)
        for seq_len in seq_len_list:
            writer.writerow([seq_len] + [backward_results[seq_len][d_model] for d_model in d_model_list])

    logging.info(f"Forward CSV: {forward_csv_path.resolve()}")
    logging.info(f"Backward CSV: {backward_csv_path.resolve()}")
