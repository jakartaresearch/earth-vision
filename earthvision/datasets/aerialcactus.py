"""Aerial Cactus Dataset from Kaggle."""
import os
import shutil
import posixpath
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Resize
from .utils import _urlretrieve, _load_img


class AerialCactus(Dataset):
    """Aerial Cactus Dataset.
    <https://www.kaggle.com/c/aerial-cactus-identification>
    """

    mirrors = "https://storage.googleapis.com/ossjr"
    resources = "cactus-aerial-photos.zip"

    def __init__(self,
                 root: str,
                 data_mode: str = 'training_set',
                 transform=Resize((32, 32)),
                 target_transform=None):

        self.root = root
        self.data_mode = data_mode
        self.transform = transform
        self.target_transform = target_transform

        if not self._check_exists():
            self.download()
            self.extract_file()

        self.img_labels = self.get_path_and_label()

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.img_labels.iloc[idx, 0]
        label = self.img_labels.iloc[idx, 1]
        image = _load_img(img_path)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        image = np.array(image)
        image = torch.from_numpy(image)
        sample = (image, label)

        return sample

    def __iter__(self):
        for index in range(self.__len__()):
            yield self.__getitem__(index)

    def get_path_and_label(self):
        """Return dataframe type consist of image path and corresponding label."""
        classes = {'cactus': 1, 'no_cactus': 0}
        image_path = []
        label = []
        for cat, enc in classes.items():
            cat_path = os.path.join(
                self.root, 'cactus-aerial-photos', self.data_mode, self.data_mode, cat)
            cat_image = [os.path.join(cat_path, path)
                         for path in os.listdir(cat_path)]
            cat_label = [enc] * len(cat_image)
            image_path += cat_image
            label += cat_label
        df = pd.DataFrame({'image': image_path, 'label': label})

        return df

    def _check_exists(self):
        self.train_path = os.path.join(
            self.root, "cactus-aerial-photos", "training_set", "training_set")
        self.test_path = os.path.join(
            self.root, "cactus-aerial-photos", "validation_set", "validation_set")

        return os.path.exists(os.path.join(self.train_path, "cactus")) and \
            os.path.exists(os.path.join(self.train_path, "no_cactus")) and \
            os.path.exists(os.path.join(self.test_path, "cactus")) and \
            os.path.exists(os.path.join(self.test_path, "no_cactus"))

    def download(self):
        """Download and extract file."""
        file_url = posixpath.join(self.mirrors, self.resources)
        _urlretrieve(file_url, os.path.join(self.root, self.resources))

    def extract_file(self):
        """Extract file from compressed."""
        shutil.unpack_archive(self.resources, self.root)
        os.remove(os.path.join(self.root, self.resources))
