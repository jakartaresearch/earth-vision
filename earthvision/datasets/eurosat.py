"""Eurosat Dataset."""
import os
import requests
import shutil
import numpy as np
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
from .utils import _urlretrieve, _load_img
import pandas as pd


class EuroSat():
    """EuroSat Land Cover Categories. 
    `Download EuroSat RGB <http://madm.dfki.de/files/sentinel>`

    Args:
        root (string): Root directory of dataset.
    """
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
    

    def __init__(self, root: str):
        self.root = root
        self.data_url = "http://madm.dfki.de/files/sentinel/EuroSAT.zip"
        self._check_exists()

        self.img_labels = self.get_path_and_label()

    def __getitem__(self, index):
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

    def __len__(self):
        return len(self.img_labels)

    def __iter__(self):
        """Iterator of the class."""
        for index in range(self.__len__()):
            yield self.__getitem__(index)

    def _check_exists(self) -> None:

        EuroSAT_path = os.path.join(self.root, "EuroSAT", "2750")

        # check if all subdirectories ("AnnualCrop", "Forest", etc) exist
        check_subdirectory = [os.path.exists(os.path.join(EuroSAT_path,cat))
                                    for cat in list(self.classes.keys())]

        # if all subdirectories exist, load the data
        if all(check_subdirectory):
            print('EuroSAT data already exist.') 
            self.load_dataset()

        # Else, download then load them (TBD: unzip and process the data)
        else:
            print('Downloading EuroSAT.zip')
            self.download()
            # Unzip and process the data here
            print('Extracting EuroSAT.zip')
            self.extract_file()


    def download(self):
        """Download file."""
        _urlretrieve(self.data_url,filename="EuroSAT.zip") 

    # TODO: Do we still need load_dataset method?
    def load_dataset(self) -> None:
        pass

    def extract_file(self):
        """Extract the .zip file"""
        target_directory = os.path.join(self.root, "EuroSAT")
        os.makedirs(target_directory, exist_ok=True)
        shutil.unpack_archive("EuroSAT.zip",target_directory)

    def get_path_and_label(self):
        """Return dataframe type consist of image path and corresponding label."""
    
        image_path = []
        label = []
        for cat, enc in self.classes.items():
            cat_path = os.path.join(
                self.root, "EuroSAT","2750", cat)
            cat_image = [os.path.join(cat_path, path)
                         for path in os.listdir(cat_path)]
            cat_label = [enc] * len(cat_image)
            image_path += cat_image
            label += cat_label
        df = pd.DataFrame({'image': image_path, 'label': label})

        return df