from pathlib import Path
from typing import List, Tuple, Any
import numpy as np
import torch
from torch.utils.data import Dataset
from .base_triplet_dataset import BaseTripletDataset


class TripletPathDataset(BaseTripletDataset):
    def _make_triplet(self, triplet_pathes: Tuple[Path, Path, Path]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return tuple(map(self._load_image, triplet_pathes))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        make_pic = lambda path: self.transform(path)
        return list(map(make_pic, self.triplets[idx]))
