from torch.utils.data import Dataset
import numpy as np
import os
from pathlib import Path
import torch
from tqdm import tqdm
import random
import cv2


class SiamseDatasetPath(Dataset):
    def __init__(self, data_path: Path, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.anchor_paths, self.positive_paths, self.negative_paths = self.make_triplets_dataset()

    def __len__(self):
        filecount = 0
        for dirpath, dirs, files in os.walk(self.data_path):
            for filename in files:
                filecount += 1
        return filecount

    def __getitem__(self, idx) -> torch.Tensor:
        return self.transform(self.load_image(self.anchor_paths[idx])), self.transform(
            self.load_image(self.positive_paths[idx])), self.transform(self.load_image(self.negative_paths[idx]))

    @staticmethod
    def load_image(path: Path) -> np.ndarray:
        image = cv2.imread(path)
        return image

    def make_triplets_dataset(self) -> list:
        anchor_paths = []
        positive_paths = []
        negative_paths = []

        for folder in tqdm(os.listdir(self.data_path)):
            cur_path = os.path.join(self.data_path, folder)

            for file in os.listdir(cur_path):
                path_anc = os.path.join(cur_path, file)
                path_pos = os.path.join(cur_path, random.choice(os.listdir(cur_path)))

                name_neg = random.choice(os.listdir(self.data_path))
                path_neg = os.path.join(self.data_path, name_neg)

                anchor_paths.append(path_anc)
                positive_paths.append(path_pos)
                negative_paths.append(path_neg + '\\' + random.choice(os.listdir(path_neg)))

        return [anchor_paths, positive_paths, negative_paths]


class SiamseDatasetPics(Dataset):
    def __init__(self, data_path: Path, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.anchor, self.positive, self.negative = self.make_triplets_dataset()

    def __len__(self):
        return len(self.anchor)

    def __getitem__(self, idx) -> torch.Tensor:
        return self.transform(self.anchor[idx]), self.transform(self.positive[idx]), self.transform(self.negative[idx])

    @staticmethod
    def load_image(path: Path) -> np.ndarray:
        image = cv2.imread(path)
        return image

    def make_triplets_dataset(self) -> list:
        anchor_images = []
        positive_images = []
        negative_images = []

        for folder in tqdm(os.listdir(self.data_path)):
            cur_path = os.path.join(self.data_path, folder)

            for file in os.listdir(cur_path):
                path_anc = os.path.join(cur_path, file)
                path_pos = os.path.join(cur_path, random.choice(os.listdir(cur_path)))

                name_neg = random.choice(os.listdir(self.data_path))
                path_neg = os.path.join(self.data_path, name_neg)

                a_image = self.load_image(path_anc)
                pos_image = self.load_image(path_pos)
                neg_image = self.load_image(path_neg + '/' + random.choice(os.listdir(path_neg)))

                anchor_images.append(a_image)
                positive_images.append(pos_image)
                negative_images.append(neg_image)

        return [np.array(anchor_images), np.array(positive_images), np.array(negative_images)]
