from pathlib import Path
from typing import List, Tuple, Any
import torch
from torch.utils.data import Dataset
from .base_triplet_dataset import BaseTripletDataset


class TripletPicsDataset(BaseTripletDataset):
    @staticmethod
    def _make_triplet(triplet_pathes: Tuple[Path, Path, Path]) -> Tuple[Any, Any, Any]:
        return triplet_pathes

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        make_pic = lambda path: self.transform(self._load_image(path))
        return list(map(make_pic, self.triplets[idx]))
