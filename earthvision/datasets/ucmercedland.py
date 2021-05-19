import sys
import os
import numpy as np
import random
import cv2
import torch

from PIL import Image
from torch.utils.data import Dataset

class UCMercedLand():
    """UC Merced Land Use Dataset.
    <http://weegee.vision.ucmerced.edu/datasets/landuse.html>
    """

    def __init__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def download(self):
        """download and extract file.
        """
        raise NotImplementedError
