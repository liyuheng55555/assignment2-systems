import logging
from timeit import default_timer

import torch
import einops
import numpy

from cs336_basics.ch3.TransformerLM import TransformerLM
from cs336_basics.ch4.cross_entropy import cross_entropy
from cs336_basics.ch4.optimizer_AdamW import AdamW


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


def main_loop(
        model: TransformerLM,
        optimizer: AdamW,
        vocab_size: int,
        context_length: int,
        batch_size: int,
        mode: int):
    input = torch.randint(0, vocab_size, (batch_size, context_length,))
    result = model.forward(input)
    if mode >= 1:
        result: torch.Tensor = einops.rearrange(result, "batch context_length vocab_size -> (batch context_length) vocab_size")
        target = torch.arange(batch_size * context_length, device=result.device)
        ce: torch.Tensor = cross_entropy(result, target)
        ce.backward()
    if mode >= 2:
        optimizer.step()
        optimizer.zero_grad()


# ADAMW_PARAMS
LEARNING_RATE = 9e-4
BETAS = (0.9, 0.999)
EPS = 1e-8
WEIGHT_DECAY = 0.01

# GRADIENT_CLIPPING
L2_NORM = 1.0

# LR_SCHEDULE
COSINE_CYCLE_ITERS = 1000

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps")

def benchmark(
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        batch_size: int,
        benchmark_loops: int,
        warm_up_loops: int,
        mode: int, # 0: forward   1: forward+backward   2: f+w+optimize
        device: torch.device = torch.device("cuda"),
        data_type: torch.dtype = torch.bfloat16
) -> list[float]:
    weights = weights_init(vocab_size, d_model, num_layers, d_ff, device, data_type)
    model = TransformerLM(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, 10000, weights)
    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=BETAS,
        weight_decay=WEIGHT_DECAY,
        eps=EPS,
        device=device,
        cosine_cycle_iters=COSINE_CYCLE_ITERS,
    )

    logging.info("warm up start")

    for i in range(warm_up_loops):
        main_loop(model, optimizer, vocab_size, context_length, batch_size, mode)

    logging.info("warm up finished")

    logging.info("benchmark start")

    result = []

    for i in range(benchmark_loops):
        start = default_timer()
        main_loop(model, optimizer, vocab_size, context_length, batch_size, mode)
        torch.cuda.synchronize()
        result.append(default_timer() - start)
        logging.info(f"loop {i} done")

    logging.info("benchmark finished")

    return result


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    hyper_parameters = {}
    hyper_parameters['small'] = {
        "d_model": 768,
        "d_ff": 3072,
        "num_layers": 12,
        "num_heads": 12,
    }
    hyper_parameters['medium'] = {
        "d_model": 1024,
        "d_ff": 4096,
        "num_layers": 24,
        "num_heads": 16,
    }
    hyper_parameters['large'] = {
        "d_model": 1280,
        "d_ff": 5120,
        "num_layers": 36,
        "num_heads": 20,
    }
    hyper_parameters['xl'] = {
        "d_model": 2560,
        "d_ff": 10240,
        "num_layers": 32,
        "num_heads": 32,
    }
    hyper_parameters['10B'] = {
        "d_model": 4608,
        "d_ff": 12288,
        "num_layers": 50,
        "num_heads": 36,
    }
    for name, params in hyper_parameters.items():
        logging.info(f"for {name}")
        result = benchmark(
            vocab_size=10000,
            context_length=512,
            d_model=params["d_model"],
            num_layers=params["num_layers"],
            num_heads=params["num_heads"],
            d_ff=params["d_ff"],
            batch_size=4,
            benchmark_loops=10,
            warm_up_loops=5,
            mode=2
        )
        print(f"[{name}] average:{numpy.average(result):.4f} std:{numpy.std(result):.4f}")
