import os
import shutil
import posixpath
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Resize
from PIL import Image
from .utils import _urlretrieve, _load_img


class EuroSat():
    """EuroSat Land Cover Categories. 
    `Download EuroSat RGB <http://madm.dfki.de/files/sentinel>`

    Args:
        root (string): Root directory of dataset.
    """
    
    mirrors = "http://madm.dfki.de/files/sentinel"
    resources = "EuroSAT.zip"
    classes = {"AnnualCrop": 0, \
                    "Forest": 1, \
                    "HerbaceousVegetation": 2, \
                    "Highway": 3, \
                    "Industrial": 4, \
                    "Pasture": 5, \
                    "PermanentCrop": 6, \
                    "Residential": 7, \
                    "River": 8, \
                    "SeaLake": 9}


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
        self.dir_classes = list(self.classes.keys())
        
        return all([os.path.exists(os.path.join(self.data_path, i)) for i in self.dir_classes])

    def download(self):
       """Download file"""
       file_url = posixpath.join(self.mirrors, self.resources)
       _urlretrieve(file_url, os.path.join(self.root, self.resources))

    def extract_file(self):
        """Extract the .zip file"""
        shutil.unpack_archive(os.path.join(self.root, self.resources), self.root)
        os.remove(os.path.join(self.root, self.resources))

    def get_path_and_label(self):
        """Return dataframe type consist of image path and corresponding label."""
        image_path = []
        label = []
        for cat, enc in self.classes.items():
            cat_path = os.path.join(
                self.root, self.data_mode, cat)
            cat_image = [os.path.join(cat_path, path)
                            for path in os.listdir(cat_path)]
            cat_label = [enc] * len(cat_image)
            image_path += cat_image
            label += cat_label
        df = pd.DataFrame({'image': image_path, 'label': label})

        return df