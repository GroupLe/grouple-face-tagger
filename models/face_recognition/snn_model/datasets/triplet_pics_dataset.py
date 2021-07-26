from pathlib import Path
from typing import Tuple
import torch
import numpy as np
from .base_triplet_dataset import BaseTripletDataset


class TripletPicsDataset(BaseTripletDataset):

    @staticmethod
    def _make_triplet( triplet_pathes: Tuple[Path, Path, Path]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return tuple(map(super()._load_image, triplet_pathes))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        make_pic = lambda path: self.transform(self._load_image(path))
        return list(map(make_pic, self.triplets[idx]))
