import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):

    def __init__(self, margin):
        super().__init__()
        self.margin = margin

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)
        distance_negative = (anchor - negative).pow(2).sum(1)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()


def accuracy(anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor, batch_size: int) -> float:
    total_correct = 0
    cos = nn.CosineSimilarity()
    a_p = cos(anchor, positive)
    a_n = cos(anchor, negative)
    for i in range(len(a_p)):
        total_correct = total_correct+1 if a_p[i] > a_n[i] else total_correct
    return total_correct/batch_size

