import os
import random
from pathlib import Path
from typing import List, Tuple, Any
from tqdm import tqdm
import numpy as np
import cv2
import torch
from multiprocessing import Pool
from torch.utils.data import Dataset


class BaseTripletDataset(Dataset):
    def __init__(self, data_path: Path, transform: torch.nn.Module = None):
        self.data_path = data_path
        self.transform = transform
        self.triplets = self._make_triplets_dataset(data_path)

    def __len__(self):
        return len(self.triplets)

    def _make_class_triplets(self, pos_folder: Path) -> List[Tuple[Any, Any, Any]]:
        """Makes triplets for given class (=folder with one class pics)"""
        triplets = []
        data_path = self.data_path

        for pos_file in os.listdir(data_path / pos_folder):
            path_anc = data_path / pos_folder / pos_file
            # sample random positive pair
            pos_samples = os.listdir(data_path / pos_folder)
            pos_samples.remove(pos_file)
            path_pos = data_path / pos_folder / random.choice(pos_samples)

            # avoid negative samples from positive pics folder
            neg_folders = os.listdir(data_path)
            neg_folders.remove(pos_folder)
            # sample negative class
            neg_folder = random.choice(neg_folders)
            # sample negative example from selected class
            neg_samples = os.listdir(data_path / neg_folder)
            neg_path = data_path / neg_folder / random.choice(neg_samples)

            triplet = self._make_triplet([path_anc, path_pos, neg_path])
            triplets.append(triplet)

        return triplets

    def _make_triplets_dataset(self, data_path: Path) -> List[Tuple[Any, Any, Any]]:
        random.seed(2021)

        files = os.listdir(data_path)

        processes = 5
        with Pool(processes) as pool:
            triplet_dataset = list(tqdm(pool.imap(self._make_class_triplets, files)))

        return triplet_dataset

    @staticmethod
    def _load_image(path: Path) -> np.ndarray:
        image = cv2.imread(str(path))
        return image

    @staticmethod
    def _make_triplet(triplet_pathes: Tuple[Path, Path, Path]) -> Tuple[Any, Any, Any]:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError

