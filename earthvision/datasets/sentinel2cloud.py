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

class Sentinel2Cloud():

    """
    Sentinel-2 Cloud Mask Catalogue dataset.
    classification_tags<https://zenodo.org/record/4172871/files/classification_tags.csv?download=1>
    subscenes<https://zenodo.org/record/4172871/files/subscenes.zip?download=1>
    masks<https://zenodo.org/record/4172871/files/masks.zip?download=1>
    """

    def __init__(self):
        raise NotImplementedError

    def __itemget__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError

    def download(self):
        raise NotImplementedError

    def _check_exists(self):
        raise NotImplementedError
