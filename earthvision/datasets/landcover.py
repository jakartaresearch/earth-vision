import os
import shutil
import posixpath
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Resize
from .utils import _urlretrieve, _load_img

class LandCover():

    """
    The LandCover.ai (Land Cover from Aerial Imagery) dataset.
    <https://landcover.ai/download/landcover.ai.v1.zip>
    """

    mirrors = "https://landcover.ai/download/"
    resources = "landcover.ai.v1.zip"

    def __init__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        img_path = self.img_labels.iloc[idx, 0]
        mask_path = self.img_labels.iloc[idx, 1]
        image = _load_img(img_path)
        mask = _load_img(mask_path)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)
        image = np.array(image)
        image = torch.from_numpy(image)
        mask = np.array(mask)
        mask = torch.from_numpy(mask)
        sample = (image, mask)

        return sample

    def __len__(self):
        return len(self.img_labels)

    def download(self):
        """download and extract file.
        """
        raise NotImplementedError


  
