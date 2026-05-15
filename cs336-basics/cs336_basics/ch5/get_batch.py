import torch
import numpy as np
import numpy.typing as npt


'''
What if the dataset is too big to load into memory? We can use a Unix system call named mmap which 
maps a file on disk to virtual memory, and lazily loads the file contents when that memory location is 
accessed. Thus, you can “pretend” you have the entire dataset in memory. Numpy implements this 
through np.memmap (or the flag mmap_mode='r' to np.load, if you originally saved the array with np.save), 
which will return a numpy array-like object that loads the entries on-demand as you access them. When 
sampling from your dataset (i.e., a numpy array) during training, be sure to load the 
dataset in memory-mapped mode (via np.memmap or the flag mmap_mode='r' to np.load, depending on 
how you saved the array). Make sure you also specify a dtype that matches the array that you’re loading. 
It may be helpful to explicitly verify that the memory-mapped data looks correct (e.g., doesn’t contain 
values beyond the expected vocabulary size).
'''
def get_batch(
        dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng()
    starts = rng.choice(dataset.size - context_length, size=batch_size, replace=False)
    data_offsets = np.arange(0, context_length)
    data_indexes = starts[:, None] + data_offsets[None, :]
    target_indexes = starts[:, None] + (data_offsets[None, :] + 1)
    return torch.tensor(dataset[data_indexes], device=device, dtype=torch.int), torch.tensor(dataset[target_indexes], device=device, dtype=torch.int)