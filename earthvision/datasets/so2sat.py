"""So2Sat Imagery"""
import os
import posixpath
import shutil
import numpy as np
import pandas as pd
import torch
import h5py
from .utils import _urlretrieve
from torch.utils.data import Dataset

class So2Sat(Dataset):
    """
    So2Sat Dataset to Predict Local Climate Zone (LCZ): <https://mediatum.ub.tum.de/1454690>
    """

    mirrors = "https://dataserv.ub.tum.de/s/m1454690/download?path=/&files="
    resources = ["training.h5", "validation.h5"]

    def __init__(self,
                 root: str,
                 data_mode: str = 'training',
    ):
        self.root = root
        self.data_mode = data_mode

        if not self._check_exists():
            self.download()

        self.img_labels = self.get_path_and_label()

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        sen1 = self.img_labels["sen1"][idx]
        sen1 = torch.from_numpy(sen1)

        sen2 = self.img_labels["sen2"][idx]
        sen2 = torch.from_numpy(sen2)

        label = self.img_labels["label"][idx]
        label = torch.from_numpy(label)

        return (sen1, sen2, label)

    def get_path_and_label(self):
        """Return dataframe type consist of image path and corresponding label."""
        file = h5py.File(os.path.join(self.root, f"{self.data_mode}.h5"), 'r')

        sen1 = np.array(file["sen1"])
        sen2 = np.array(file["sen2"])
        label = np.array(file["label"])

        return {"sen1": sen1, "sen2": sen2, "label": label}

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.resources[0])) and \
            os.path.exists(os.path.join(self.root, self.resources[1]))

    def download(self):
        """Download and extract file."""
        if not os.path.exists(self.root):
            os.makedirs(self.root)

        for resource in self.resources:
            file_url = posixpath.join(self.mirrors, resource)
            _urlretrieve(file_url, os.path.join(self.root, resource)) 