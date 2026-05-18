import torch


def softmax(x: torch.Tensor, i: int, temp: float = 1) -> torch.Tensor:
    max_i = x.max(dim=i, keepdim=True)
    stabilized_x = x - max_i.values
    if temp != 1:
        temp_x = stabilized_x / temp
        temp_x = temp_x.to(dtype=x.dtype)
    else:
        temp_x = x
    exp = torch.exp(temp_x)
    exp_sum = exp.sum(dim=i, keepdim=True)
    return exp / exp_sum