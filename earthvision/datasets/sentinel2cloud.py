import os
import shutil
import posixpath
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Resize
from .utils import _urlretrieve, _load_img, _load_npy

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
            self.extract_file()
        
        self.img_labels = ""
        

    def __itemget__(self, idx):
        img_path = self.img_labels.iloc[idx, 0]
        mask_path = self. img_labels.iloc[idx, 1]

        image = _load_npy(img_path)
        mask = _load_npy(mask_path)
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

    def get_image_path_and_mask_path(self):
        """Return dataframe type consist of image path and mask path."""
        image_path = []
        mask_path = []

        img_path = os.path.join(self.root, 'sentinel2cloud', 'subscenes')
        msk_path = os.path.join(self.root, 'sentinel2cloud', 'masks')

        images_path = [os.path.join(img_path, path)
                       for path in os.listdir(img_path)]
        images_path.sort()
        masks_path = [os.path.join(img_path, path)
                      for path in os.listdir(msk_path)]
        masks_path.sort()

        df = pd.DataFrame({'image': images_path, 'mask': masks_path})
        return df

    def download(self):
        """download and extract file.
        """
        file_url = posixpath.join(self.mirrors, self.resources)
        _urlretrieve(file_url, os.path.join(self.root, self.resources))

        mask_file_url = posixpath.join(self.mirrors, self.mask_resources)
        _urlretrieve(mask_file_url, os.path.join(self.root, self.mask_resources))
        
        shapefile_file_url = posixpath.join(self.mirrors, self.shapefile_resources)
        _urlretrieve(shapefile_file_url, os.path.join(self.root, self.shapefile_resources))

        thumbnail_file_url = posixpath.join(self.mirrors, self.thumbnail_resources)
        _urlretrieve(thumbnail_file_url, os.path.join(self.root, self.thumbnail_resources))
        
        altmasks_file_url = posixpath.join(self.mirrors, self.alt_mask_resources)
        _urlretrieve(altmasks_file_url, os.path.join(self.root, self.alt_mask_resources))


    def _check_exists(self):
        """ Check file has been download or not
        """
        self.data_path = os.path.join(self.root, "sentinel2cloud")

        return os.path.exists(os.path.join(self.data_path, "subscenes")) and \
            os.path.exists(os.path.join(self.data_path, "masks")) and \
            os.path.exists(os.path.join(self.data_path, "shapefiles")) and \
            os.path.exists(os.path.join(self.data_path, "thumbnails")) and \
            os.path.exists(os.path.join(self.data_path, "alt_masks"))

    def extract_file(self):
        """Extract file from compressed.
        """
        
        os.makedirs(os.path.join(self.root, "sentinel2cloud"))

        shutil.unpack_archive(os.path.join(
            self.root, self.resources), os.path.join(self.root, "sentinel2cloud"))
        os.remove(os.path.join(self.root, self.resources))

        shutil.unpack_archive(os.path.join(
            self.root, self.mask_resources), os.path.join(self.root, "sentinel2cloud"))
        os.remove(os.path.join(self.root, self.mask_resources))

        shutil.unpack_archive(os.path.join(
            self.root, self.shapefile_resources), os.path.join(self.root, "sentinel2cloud"))
        os.remove(os.path.join(self.root, self.shapefile_resources))

        shutil.unpack_archive(os.path.join(
            self.root, self.thumbnail_resources), os.path.join(self.root, "sentinel2cloud"))
        os.remove(os.path.join(self.root, self.thumbnail_resources))

        shutil.unpack_archive(os.path.join(
            self.root, self.alt_mask_resources), os.path.join(self.root, "sentinel2cloud"))
        os.remove(os.path.join(self.root, self.alt_mask_resources))