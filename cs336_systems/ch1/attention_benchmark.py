import logging
import time
from pathlib import Path
import csv

import torch

from cs336_basics.ch3.MultiHeadAttention import MultiHeadAttentionWithRope, SingleHeadAttentionWithRope

def attention_benchmark(
        d_model: int,
        seq_len: int,
        output_dir: Path,
        device: torch.device = torch.device("cuda"),
        dtype: torch.dtype = torch.bfloat16
) -> (float, float, float):
    try:
        logging.info(f"for d_model={d_model}, seq_len={seq_len}")
        logging.info(f"init model")
        model = SingleHeadAttentionWithRope(
            d_model,
            seq_len,
            10000,
            torch.empty(d_model, d_model, device=device, dtype=dtype).normal_(mean=0.0, std=0.02),
            torch.empty(d_model, d_model, device=device, dtype=dtype).normal_(mean=0.0, std=0.02),
            torch.empty(d_model, d_model, device=device, dtype=dtype).normal_(mean=0.0, std=0.02),
            torch.empty(d_model, d_model, device=device, dtype=dtype).normal_(mean=0.0, std=0.02),
        )
        model = torch.compile(model)
        model.to(device=device)

        x = torch.rand((8, seq_len, d_model), device=device, dtype=dtype)
        token_positions = torch.arange(0, seq_len, device=device)

        for i in range(5):
            y = model.forward(x, token_positions)
            y.sum().backward()
            model.zero_grad(set_to_none=True)

        # pack_sum = 0
        # seen_store = set()

        # def pack_hook(t):
        #     nonlocal pack_sum
        #     if t.data_ptr() not in seen_store:
        #         seen_store.add(t.data_ptr())
        #         numel, shape, dtype, grad_fn = t.numel(), t.shape, t.dtype, t.grad_fn
        #         pack_sum += numel
        #         print(f"Saving residual: {numel=} {shape=}, {dtype=}, {grad_fn=}")
        #     return t
        
        # def unpack_hook(t):
        #     return t
        
        # with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
        #     model.forward(x, token_positions)
        #     logging.info(f"{pack_sum=}")
        #     return

        forward_time = 0
        backward_time = 0
        memory = 0
        logging.info("start")
        for i in range(100):
            model.zero_grad(set_to_none=True)

            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

            forward_start = time.perf_counter()
            y = model.forward(x, token_positions)
            torch.cuda.synchronize()
            forward_time += time.perf_counter() - forward_start

            memory += torch.cuda.memory_allocated()
            
            backward_start = time.perf_counter()
            y.sum().backward()
            torch.cuda.synchronize()
            backward_time += time.perf_counter() - backward_start

            if i % 10 == 0:
                logging.info(f"{i}")

        torch.cuda.empty_cache()

        return forward_time, backward_time, memory / 100
    
    except torch.cuda.OutOfMemoryError as e:
        logging.info("cuda out of memory")
        torch.cuda.empty_cache()
        return -1, -1, -1

if __name__ == "__main__":
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    d_model_list = [
        16, 32, 
        64, 
        128
    ]
    seq_len_list = [
        256, 1024, 
        4096, 
        8192, 
        16384
    ]
    

    test_dir = Path("attention_benchmark_results") / time.strftime("%Y%m%d_%H%M%S")
    test_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Saving benchmark outputs to: {test_dir.resolve()}")

    forward_results = {seq_len: {} for seq_len in seq_len_list}
    backward_results = {seq_len: {} for seq_len in seq_len_list}
    memory_results = {seq_len: {} for seq_len in seq_len_list}

    for d_model in d_model_list:
        for seq_len in seq_len_list:
            forward_time, backward_time, memory = attention_benchmark(d_model, seq_len, output_dir=test_dir)
            forward_results[seq_len][d_model] = forward_time
            backward_results[seq_len][d_model] = backward_time
            memory_results[seq_len][d_model] = memory

    forward_csv_path = test_dir / "time_forward.csv"
    backward_csv_path = test_dir / "time_backward.csv"
    memory_csv_path = test_dir / "memory_peak_forward.csv"

    with forward_csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seq_len"] + d_model_list)
        for seq_len in seq_len_list:
            writer.writerow([seq_len] + [f"{forward_results[seq_len][d_model]:.2f} s" for d_model in d_model_list])

    with backward_csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seq_len"] + d_model_list)
        for seq_len in seq_len_list:
            writer.writerow([seq_len] + [f"{backward_results[seq_len][d_model]:.2f} s" for d_model in d_model_list])

    with memory_csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seq_len"] + d_model_list)
        for seq_len in seq_len_list:
            writer.writerow([seq_len] + [f"{memory_results[seq_len][d_model]/1024/1024:.2f} MB" for d_model in d_model_list])

    logging.info(f"Forward CSV: {forward_csv_path.resolve()}")
    logging.info(f"Backward CSV: {backward_csv_path.resolve()}")
