"""Eurosat Dataset."""
import os
import requests
import shutil
import numpy as np
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
from .utils import _urlretrieve


class EuroSat():
    """EuroSat Land Cover Categories. 
    `Download EuroSat RGB <http://madm.dfki.de/files/sentinel>`

    Args:
        root (string): Root directory of dataset.
    """

    labels = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
                "Industrial", "Pasture", "PermanentCrop", "Residential",
                "River", "SeaLake"]

    def __init__(self, root: str):
        self.root = root
        self.data_url = "http://madm.dfki.de/files/sentinel/EuroSAT.zip"
        self._check_exists()

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __iter__(self):
        """Iterator of the class."""
        raise NotImplementedError

    def _check_exists(self) -> None:

        EuroSAT_path = os.path.join(self.root, "EuroSAT")

        # check if all subdirectories ("AnnualCrop", "Forest", etc) exist
        check_subdirectory = [os.path.exists(os.path.join(EuroSAT_path,label))
                                    for label in self.labels]

        # if all subdirectories exist, load the data
        if all(check_subdirectory):
            print('EuroSAT data already exist.') 
            self.load_dataset()

        # Else, download then load them (TBD: unzip and process the data)
        else:
            # with requests.get(self.data_url, stream=True) as r:
            #     with open(self.root, 'wb') as f:
            #         shutil.copyfileobj(r.raw, f)

            print('Downloading EuroSAT.zip')
            self.download()
            # Unzip and process the data here
            print('Extracting EuroSAT.zip')
            self.extract_file()


    #def download(self):
    #    """Download file."""
    #    _urlretrieve(self.data_url,filename="EuroSAT.zip") 

    def load_dataset(self) -> None:
        pass


    def extract_file(self):
        """Extract the .zip file"""
        target_directory = os.path.join(self.root, "EuroSAT")
        os.makedirs(target_directory, exist_ok=True)
        shutil.unpack_archive("EuroSAT.zip",target_directory)
