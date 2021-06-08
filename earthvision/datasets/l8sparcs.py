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

class L8SPARCS():
    """Landsat 8 SPARCS Cloud. 
    <https://www.usgs.gov/core-science-systems/nli/landsat/spatial-procedures-automated-removal-cloud-and-shadow-sparcs>

    Download: <https://landsat.usgs.gov/cloud-validation/sparcs/l8cloudmasks.zip>

    Args:
        root (string): Root directory of dataset.
    """
    
    mirrors = "https://landsat.usgs.gov/cloud-validation/sparcs/"
    resources = "l8cloudmasks.zip"
    
    def __init__(self,
                root: str,
                data_mode: str = 'sending',
                transform=Resize((64,64)),
                target_transform=None):
        
        self.root = root
        self.data_mode = data_mode
        self.transform = transform
        self.target_transform = target_transform

        if not self._check_exists():
            self.download()
            self.extract_file()

    def _check_exists(self) -> None:
        self.data_path = os.path.join(
            self.root, self.data_mode)
        return os.path.exists(self.data_path)

    def download(self):
       """Download file"""
       file_url = posixpath.join(self.mirrors, self.resources)
       _urlretrieve(file_url, os.path.join(self.root, self.resources))

    def extract_file(self):
        """Extract the .zip file"""
        shutil.unpack_archive(os.path.join(self.root, self.resources), self.root)
        os.remove(os.path.join(self.root, self.resources))