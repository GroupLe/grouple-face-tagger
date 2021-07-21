import torch


class EmptyTransformation(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return img