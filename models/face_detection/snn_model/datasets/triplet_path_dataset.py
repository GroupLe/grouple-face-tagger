from pathlib import Path
from typing import Tuple, Any
import torch
from .base_triplet_dataset import BaseTripletDataset


class TripletPathDataset(BaseTripletDataset):

    @staticmethod
    def _make_triplet(triplet_pathes: Tuple[Path, Path, Path]) -> Tuple[Any, Any, Any]:
        return triplet_pathes

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        make_pic = lambda path: self.transform(path)
        return list(map(make_pic, self.triplets[idx]))
