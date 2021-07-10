from torch.utils.data import Dataset
import numpy as np
import torch
from . import functions


class TokensDataset(Dataset):
    def __init__(self, X, Y):
        self.X = self.encode_x(X)
        self.y = Y

    @staticmethod
    def encode_x(x: list) -> list:
        max_len = len(max(x, key=lambda i: len(i)))
        encoded = []
        for i in x:
            encoded.append(np.array(functions.encode(i, max_len)))
        return encoded

    @staticmethod
    def collate_fn(objs: list) -> (torch.LongTensor, torch.Tensor):
        data = ([i[0] for i in objs])
        labels = ([i[1] for i in objs])
        data = torch.LongTensor(data)
        labels = torch.tensor(labels)
        return data, labels

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
