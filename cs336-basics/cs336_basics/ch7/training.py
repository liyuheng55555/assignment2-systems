import logging
import time
import csv
from pathlib import Path
from datetime import datetime

import einops
import torch
import numpy as np
from ch3.transformer_accounting import calculate_parameters
from jaxtyping import Float
from torch import Tensor

from ch2.Tokenizer import Tokenizer
from ch3.Softmax import softmax
from ch3.TransformerLM import TransformerLM
from ch4.cross_entropy import cross_entropy
from ch4.gradient_clipping import gradient_clipping
from ch4.optimizer_AdamW import AdamW
from ch5.checkpoint import save_checkpoint, load_checkpoint
from ch5.get_batch import get_batch

############## Settings ##############

TOTAL_STEPS = 5000
BATCH_SIZE = 64

# Model Size

VOCAB_SIZE = 10000
CONTEXT_LENGTH = 256
D_MODEL = 512
D_FF = 1344
ROPE_THETA = 10000
NUM_HEADS = 16
NUM_LAYERS = 4

# ADAMW_PARAMS
LEARNING_RATE = 12e-4
BETAS = (0.9, 0.999)
EPS = 1e-8
WEIGHT_DECAY = 0.01

# GRADIENT_CLIPPING
L2_NORM = 1.0

# LR_SCHEDULE
COSINE_CYCLE_ITERS = TOTAL_STEPS

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps")
BACKEND = torch.cuda if torch.cuda.is_available() else torch.mps

DATA_TYPE = torch.bfloat16

PROFILE = False

if PROFILE:
    logging.warning("Profiling is on")

torch.manual_seed(69)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

################ Settings Done #################

weights = {
    'token_embeddings.weight': torch.empty(VOCAB_SIZE, D_MODEL, device=DEVICE, dtype=DATA_TYPE).normal_(mean=0.0, std=0.02),
    'ln_final.weight': torch.ones(D_MODEL, device=DEVICE, dtype=DATA_TYPE),
    'lm_head.weight': torch.empty(VOCAB_SIZE, D_MODEL, device=DEVICE, dtype=DATA_TYPE).normal_(mean=0, std=0.02),
}
for i in range(NUM_LAYERS):
    layer_weights = {
        f'layers.{i}.attn.q_proj.weight': torch.empty(D_MODEL, D_MODEL, device=DEVICE, dtype=DATA_TYPE).normal_(mean=0.0, std=0.02),
        f'layers.{i}.attn.k_proj.weight': torch.empty(D_MODEL, D_MODEL, device=DEVICE, dtype=DATA_TYPE).normal_(mean=0.0, std=0.02),
        f'layers.{i}.attn.v_proj.weight': torch.empty(D_MODEL, D_MODEL, device=DEVICE, dtype=DATA_TYPE).normal_(mean=0.0, std=0.02),
        f'layers.{i}.attn.output_proj.weight': torch.empty(D_MODEL, D_MODEL, device=DEVICE, dtype=DATA_TYPE).normal_(mean=0.0, std=0.02),
        f'layers.{i}.ffn.w1.weight': torch.empty(D_FF, D_MODEL, device=DEVICE, dtype=DATA_TYPE).normal_(mean=0.0, std=0.02),
        f'layers.{i}.ffn.w2.weight': torch.empty(D_MODEL, D_FF, device=DEVICE, dtype=DATA_TYPE).normal_(mean=0.0, std=0.02),
        f'layers.{i}.ffn.w3.weight': torch.empty(D_FF, D_MODEL, device=DEVICE, dtype=DATA_TYPE).normal_(mean=0.0, std=0.02),
        f'layers.{i}.ln1.weight': torch.ones(D_MODEL, device=DEVICE, dtype=DATA_TYPE),
        f'layers.{i}.ln2.weight': torch.ones(D_MODEL, device=DEVICE, dtype=DATA_TYPE),
    }
    weights |= layer_weights



logging.info("Init model...")
model = TransformerLM(
    VOCAB_SIZE,
    CONTEXT_LENGTH,
    D_MODEL,
    NUM_LAYERS,
    NUM_HEADS,
    D_FF,
    ROPE_THETA,
    weights
)

logging.info("Init optimizer...")
optimizer = AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    betas=BETAS,
    weight_decay=WEIGHT_DECAY,
    eps=EPS,
    device=DEVICE,
    cosine_cycle_iters=COSINE_CYCLE_ITERS,
)

# for param in model.parameters():
#     print(param.shape)

def train(checkpoint_path: Path = None):
    logging.info("training start")
    data_path = Path("/data/cs336/data/tinystories_train_tokenized/result.npy")
    data = np.load(data_path, mmap_mode="r")

    train_start_time = datetime.now()
    run_id = train_start_time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path("training_runs") / f"run_{run_id}"
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    csv_file_path = run_dir / f"train_log_{run_id}.csv"
    logging.info(f"training run dir: {run_dir}")
    csv_file = open(csv_file_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["record_type", "timestamp", "key", "value", "step", "loss"])

    hyperparameters = {
        "total_steps": TOTAL_STEPS,
        "batch_size": BATCH_SIZE,
        "vocab_size": VOCAB_SIZE,
        "context_length": CONTEXT_LENGTH,
        "d_model": D_MODEL,
        "d_ff": D_FF,
        "rope_theta": ROPE_THETA,
        "num_heads": NUM_HEADS,
        "num_layers": NUM_LAYERS,
        "learning_rate": LEARNING_RATE,
        "betas": BETAS,
        "eps": EPS,
        "weight_decay": WEIGHT_DECAY,
        "l2_norm": L2_NORM,
        "cosine_cycle_iters": COSINE_CYCLE_ITERS,
        "device": str(DEVICE),
        "data_type": str(DATA_TYPE),
    }
    for key, value in hyperparameters.items():
        csv_writer.writerow(["hyperparameter", train_start_time.isoformat(), key, value, "", ""])
    csv_file.flush()
    logging.info(f"training csv log: {csv_file_path}")

    start_iteration = 0
    if checkpoint_path is not None:
        start_iteration = load_checkpoint(checkpoint_path, model, optimizer)

    start_time = time.perf_counter()
    loss_sum = 0
    for iteration in range(start_iteration, TOTAL_STEPS):
        batch, target = get_batch(data, batch_size=BATCH_SIZE, context_length=CONTEXT_LENGTH, device=DEVICE)

        if PROFILE:
            BACKEND.synchronize()
            t_forward = time.perf_counter()
        result = model.forward(batch.long())
        if PROFILE:
            BACKEND.synchronize()
            logging.info(f"forward: {time.perf_counter() - t_forward:.4f}s")

        # entropy
        if PROFILE:
            BACKEND.synchronize()
            t_entropy = time.perf_counter()
        result = einops.rearrange(result, "batch context_length vocab_size -> (batch context_length) vocab_size")
        target1 = einops.rearrange(target, "batch context_length -> (batch context_length)")
        entropy: torch.Tensor = cross_entropy(result, target1)
        if PROFILE:
            BACKEND.synchronize()
            logging.info(f"cross entropy: {time.perf_counter() - t_entropy:.4f}s")

        if PROFILE:
            BACKEND.synchronize()
            t_backward = time.perf_counter()
        entropy.backward()
        if PROFILE:
            BACKEND.synchronize()
            logging.info(f"backward: {time.perf_counter() - t_backward:.4f}s")

        gradient_clipping(model.parameters(), L2_NORM)

        if PROFILE:
            BACKEND.synchronize()
            t_optimize = time.perf_counter()
        optimizer.step()
        if PROFILE:
            BACKEND.synchronize()
            logging.info(f"optimize: {time.perf_counter() - t_optimize:.4f}s")

        optimizer.zero_grad()
        if iteration % 1000 == 0:
            BACKEND.synchronize()
            logging.info("saving checkpoint...")
            ckpt_path = checkpoint_dir/f"{iteration}.ckpt"
            save_checkpoint(model, optimizer, iteration, ckpt_path)
            logging.info(f"checkpoint {ckpt_path.__str__()} saved")
        if iteration % 100 == 0:
            BACKEND.synchronize()
            t = time.perf_counter() - start_time
            logging.info(f"last 100 iterations:  {t:.4f}s  average_loss: {loss_sum / 10:.4f}")
            loss_sum = 0
            start_time = time.perf_counter()
        if iteration % 10 == 0:
            logging.info(f"iteration: {iteration:06d}  loss: {entropy.item()}")
            loss_sum += entropy.item()
            csv_writer.writerow(["metric", datetime.now().isoformat(), "", "", iteration, entropy.item()])
            csv_file.flush()


    # BACKEND.synchronize()
    csv_file.close()

    save_checkpoint(model, optimizer, TOTAL_STEPS, checkpoint_dir/f"{TOTAL_STEPS}.ckpt")


def decode(output: Float[Tensor, "context_length vocab_size"], vocab: list[bytes]):
    useful = output[-1]
    probability = softmax(useful, -1)
    choice = torch.multinomial(probability, num_samples=1).item()
    return vocab[choice]


def infer():
    checkpoint_dir = Path("/data/cs336/cs336-assignment1-basics/training_runs/run_20260513_163104_batch64_3090/checkpoints")
    data_path = Path("/data/cs336/data/tinystories_train_tokenized/result.npy")
    data = np.load(data_path, mmap_mode="r")
    ckpt_path = checkpoint_dir/"5000.ckpt"
    load_checkpoint(ckpt_path, model, optimizer)
    batch, _ = get_batch(data, batch_size=1, context_length=CONTEXT_LENGTH, device=DEVICE)

    tokenizer_file_path = "/data/cs336/data/TinyStoriesV2-GPT4-train.json"
    tokenizer = Tokenizer.from_json(tokenizer_file_path)
    seed_ids = batch[0].tolist()
    print(tokenizer.decode(seed_ids), end="", flush=True)

    print("开始推理！")

    model.eval()
    context = batch.long()
    with torch.no_grad():
        try:
            while True:
                output = model.forward(context)
                next_logits = output[0, -1]
                probability = softmax(next_logits, -1, temp=0.5)
                next_id = torch.multinomial(probability, num_samples=1)
                context = torch.cat([context, next_id.view(1, 1)], dim=1)[:, -CONTEXT_LENGTH:]
                print(tokenizer.decode([next_id.item()]), end="", flush=True)
        except KeyboardInterrupt:
            print()


def accounting():
    calculate_parameters(VOCAB_SIZE, CONTEXT_LENGTH, NUM_LAYERS, D_MODEL, NUM_HEADS, D_FF)


# train(checkpoint_path=Path("checkpoints/1000.ckpt"))
# infer()
if __name__ == "__main__":
    accounting()
    # train()
    infer()
    
