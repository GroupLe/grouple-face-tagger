import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor, size_average=True):  # с нормой, размерности
        distance_positive = (anchor - positive).pow(2).sum(1)
        distance_negative = (anchor - negative).pow(2).sum(1)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()


def accuracy(anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor, batch_size: int):
    total_correct = 0
    cos = nn.CosineSimilarity()
    a_p = cos(anchor, positive)
    a_n = cos(anchor, negative)
    for i in range(len(a_p) - 1):
        total_correct = total_correct+1 if a_p[i] > a_n[i] else total_correct
    return total_correct/batch_size

# def triplet_loss(anchor, positive, negative):
#     a_p = cos(anchor, positive)
#     a_n = cos(anchor, negative)
#     norma = np.linalg.norm
#     loss = max((norma(a_p.detach().numpy()) - norma(a_n.detach().numpy())) + margin, 0)
#     #print(loss.size())
#     return loss
