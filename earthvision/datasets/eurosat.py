import os
import shutil
import posixpath
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Resize
from .utils import _urlretrieve, _load_img


class EuroSat():
    """EuroSat Land Cover Categories. 
    `Download EuroSat RGB <http://madm.dfki.de/files/sentinel>`

    Args:
        root (string): Root directory of dataset.
    """

    mirrors = "http://madm.dfki.de/files/sentinel"
    resources = "EuroSAT.zip"

    labels = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
                "Industrial", "Pasture", "PermanentCrop", "Residential",
                "River", "SeaLake"]

    def __init__(self,
                root: str,
                data_mode: str = '2750',
                transform=Resize((64, 64)),
                target_transform=None):
                
        self.root = root
        self.data_mode = data_mode
        self.transform = transform
        self.target_transform = target_transform

        if not self._check_exists():
            self.download()
            self.extract_file()

        self.img_labels = self.get_path_and_label()


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

    def __len__(self):
        return len(self.img_labels)

    def __iter__(self):
        for index in range(self.__len__()):
            yield self.__getitem__(index)

    def _check_exists(self) -> None:
        self.data_path = os.path.join(
            self.root, self.data_mode)

        return os.path.exists(os.path.join(self.data_path, "AnnualCrop")) and \
            os.path.exists(os.path.join(self.data_path, "Forest")) and \
            os.path.exists(os.path.join(self.data_path, "HerbaceousVegetation")) and \
            os.path.exists(os.path.join(self.data_path, "Highway")) and \
            os.path.exists(os.path.join(self.data_path, "Industrial")) and \
            os.path.exists(os.path.join(self.data_path, "Pasture")) and \
            os.path.exists(os.path.join(self.data_path, "PermanentCrop")) and \
            os.path.exists(os.path.join(self.data_path, "Residential")) and \
            os.path.exists(os.path.join(self.data_path, "River")) and \
            os.path.exists(os.path.join(self.data_path, "SeaLake"))


    def download(self):
       """Download file"""
       file_url = posixpath.join(self.mirrors, self.resources)
       _urlretrieve(file_url, os.path.join(self.root, self.resources))


    def extract_file(self):
        """Extract the .zip file"""
        shutil.unpack_archive(os.path.join(self.root, self.resources), self.root)
        os.remove(os.path.join(self.root, self.resources))
