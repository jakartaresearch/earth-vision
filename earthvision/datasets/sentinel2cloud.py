import os
import shutil
import posixpath
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Resize
from .utils import _urlretrieve, _load_img

import glob
import cv2

class Sentinel2Cloud(Dataset):

    """
    Sentinel-2 Cloud Mask Catalogue dataset.
    classification_tags<https://zenodo.org/record/4172871/files/classification_tags.csv?download=1>
    subscenes<https://zenodo.org/record/4172871/files/subscenes.zip?download=1>
    masks<https://zenodo.org/record/4172871/files/masks.zip?download=1>
    """

    mirrors = "https://storage.googleapis.com/ossjr/sentinel2"
    resources = "subscenes.zip"

    mask_resources = "masks.zip"
    shapefile_resources = "shapefiles.zip"
    thumbnail_resources = "thumbnails.zip"
    alt_mask_resources = "alt_masks.zip"

    def __init__(self,
                 root: str,
                 data_mode: str = 'Images',
                 transform=Resize((256, 256)),
                 target_transform = None):
        
        self.root = root
        self.data_mode = data_mode
        self.transform = transform
        self.target_transform = target_transform

        if not self._check_exists():
            self.download()
            self.extract_file
        
        self.img_labels = self.download()

    def __itemget__(self, idx):
        img_path = self.img_labels.iloc[idx, 0]
        mask_path = self. img_labels.iloc[idx, 1]

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

    def __iter__(self):
        for index in range(self.__len__()):
            yield self.__itemget__(index)

    def download(self):
        """download and extract file.
        """
        file_url = posixpath.join(self.mirrors, self.resources)
        _urlretrieve(file_url, os.path.join(self.root, self.resources))
        _urlretrieve(file_url, os.path.join(self.root, self.mask_resources))
        _urlretrieve(file_url, os.path.join(self.root, self.shapefile_resources))
        _urlretrieve(file_url, os.path.join(self.root, self.thumbnail_resources))
        _urlretrieve(file_url, os.path.join(self.root, self.alt_mask_resources))


    def _check_exists(self):
        """ Check file has been download or not
        """
        self.data_path = os.path.join(self.root, "sentinel2cloud", )

        return os.path.exists(os.path.join(self.data_path, "subscenes")) and \
            os.path.exists(os.path.join(self.data_path, "masks")) and \
            os.path.exists(os.path.join(self.data_path, "shapefiles")) and \
            os.path.exists(os.path.join(self.data_path, "thumbnails")) and \
            os.path.exists(os.path.join(self.data_path, "alt_masks"))
