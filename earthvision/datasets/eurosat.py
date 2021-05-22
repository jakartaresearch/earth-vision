"""Eurosat Dataset."""
import os
import requests
import shutil
import numpy as np
import random
import torch
from torch.utils.data import Dataset
from PIL import Image


class EuroSat():
    """EuroSat Land Cover Categories. 
    `Download EuroSat RGB <http://madm.dfki.de/files/sentinel>`

    Args:
        root (string): Root directory of dataset.
    """

    def __init__(self, root: str):
        self.root = root
        self.data_url = "http://madm.dfki.de/files/sentinel/EuroSAT.zip"
        self._check_exists()

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def _check_exists(self) -> None:
        # If the files exist, load them (TBD)
        if os.path.exists(self.root):
            pass

        # Else, download then load them (TBD: unzip and process the data)
        else:
            with requests.get(self.data_url, stream=True) as r:
                with open(self.root, 'wb') as f:
                    shutil.copyfileobj(r.raw, f)

            # Unzip and process the data here

        self.load_dataset()

    def download(self):
        """Download file."""
        raise NotImplementedError

    def load_dataset(self) -> None:
        pass
