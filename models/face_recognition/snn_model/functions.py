import torch
import torch.nn as nn


class DistanceCounter:
    def __init__(self, target_emb: torch.Tensor, func):
        self.target_emb = target_emb.detach()
        self.func = func

    def count_distance(self, emb: torch.Tensor) -> torch.Tensor:
        dist = self.func(self.target_emb.unsqueeze(0), emb.unsqueeze(0))
        return dist.detach()


def accuracy(anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> float:
    total_correct = 0
    batch_size = anchor.size(0)
    # cos = nn.CosineSimilarity()
    # a_p = cos(anchor, positive)
    # a_n = cos(anchor, negative)
    # for i in range(len(a_p)):
    #     total_correct = total_correct+1 if a_p[i] > a_n[i] else total_correct
    for i in range(len(anchor)):
        a_p = (anchor[i] - positive[i]).norm()
        a_n = (anchor[i] - negative[i]).norm()
        total_correct = total_correct+1 if a_p < a_n else total_correct
    return total_correct/batch_size

