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


class VisDrone():
    """VisDrone Aerial Object Detection Dataset
        https://github.com/VisDrone/VisDrone-Dataset


    Args:
        root (string): Root directory of dataset.
    """
    
    download_command = "wget --load-cookies /tmp/cookies.txt 'https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1a2oHjcEcwXP8oUF95qiwrqzACb2YlUhn' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1a2oHjcEcwXP8oUF95qiwrqzACb2YlUhn' -O VisDrone2019-DET-train.zip && rm -rf /tmp/cookies.txt"
    "wget 'https://docs.google.com/uc?export=download&confirm=&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1a2oHjcEcwXP8oUF95qiwrqzACb2YlUhn' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1a2oHjcEcwXP8oUF95qiwrqzACb2YlUhn' -O VisDrone2019-DET-train.zip"

    def __init__(self,
                root: str):
    

        if not self._check_exists():
            self.download()
            self.extract_file()

        self.img_labels = self.get_path_and_label()

    def __getitem__(self, idx):


    def __len__(self):


    def __iter__(self):


    def _check_exists(self) -> None:


    def download(self):
       """Download file"""

    def extract_file(self):
        """Extract the .zip file"""
        

    def get_path_and_label(self):
        """Return dataframe type consist of image path and corresponding label."""
        
