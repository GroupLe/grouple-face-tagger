import torch
import torch.nn as nn


def accuracy(anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor, batch_size: int) -> float:
    total_correct = 0
    cos = nn.CosineSimilarity()
    a_p = cos(anchor, positive)
    a_n = cos(anchor, negative)
    for i in range(len(a_p)):
        total_correct = total_correct+1 if a_p[i] > a_n[i] else total_correct
    return total_correct/batch_size

