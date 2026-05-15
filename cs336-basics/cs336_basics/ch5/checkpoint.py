import logging
import os
from typing import BinaryIO, IO

import torch


def save_checkpoint(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        iteration: int,
        out: str | os.PathLike | BinaryIO | IO[bytes],
):
    data = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'iteration': iteration
    }
    torch.save(data, out)


def load_checkpoint(
        src: str | os.PathLike | BinaryIO | IO[bytes],
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
):
    logging.info("loading checkpoint...")
    data = torch.load(src)
    assert isinstance(data, dict)
    model.load_state_dict(data['model'])
    optimizer.load_state_dict(data['optimizer'])
    logging.info("checkpoint loaded")
    return data['iteration']
