import os
import shutil
import posixpath
import numpy as np
import pandas as pd
import glob
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Resize
from PIL import Image
from .utils import _urlretrieve, _load_img

class XView():
    """
    Samples at:
    https://storage.googleapis.com/ossjr/xview/train_images.tgz
    https://storage.googleapis.com/ossjr/xview/train_labels.tgz
    https://storage.googleapis.com/ossjr/xview/validation_images.tgz
    """

    mirrors = "https://landsat.usgs.gov/cloud-validation/sparcs/"
    resources = "l8cloudmasks.zip"
    
    def __init__(self,
                root: str):

        """TODO"""

        return
        

    def _check_exists(self) -> None:
        """TODO"""
        return

    def download(self):
       """Download file"""
       file_url = posixpath.join(self.mirrors, self.resources)
       _urlretrieve(file_url, os.path.join(self.root, self.resources))


    def extract_file(self):
        """Extract the .zip file"""
        shutil.unpack_archive(os.path.join(self.root, self.resources), self.root)
        os.remove(os.path.join(self.root, self.resources))


    def get_path_and_label(self):
        """Get the path of the images and labels (masks) in a dataframe"""
        """TODO"""

        return
    
    def __getitem__(self, idx):
        """Return a tensor image and its tensor mask"""
        img_path = self.img_labels.iloc[idx, 0]
        mask_path = self.img_labels.iloc[idx, 1]

        image = _load_img(img_path)
        image = np.array(image)
        image = torch.from_numpy(image)
        
        mask = _load_img(mask_path)
        mask = np.array(mask)
        mask = torch.from_numpy(mask)
        
        sample = (image, mask)

        return sample


    def __len__(self):
        return len(self.img_labels)
        

    def __iter__(self):
        for index in range(self.__len__()):
            yield self.__getitem__(index)