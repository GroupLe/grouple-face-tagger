import torch


class EmptyTransformation(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, img):
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)
