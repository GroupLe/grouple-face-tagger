import torch
import torch.nn as nn


def accuracy(anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> float:
    total_correct = 0
    batch_size = anchor.size(0)
    cos = nn.CosineSimilarity()
    a_p = cos(anchor, positive)
    a_n = cos(anchor, negative)
    for i in range(len(a_p)):
        total_correct = total_correct+1 if a_p[i] > a_n[i] else total_correct
    return total_correct/batch_size

