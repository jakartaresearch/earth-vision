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

class L7Irish():
    """Landsat 7 Irish Cloud. 
    <https://landsat.usgs.gov/landsat-7-cloud-cover-assessment-validation-data>

    Need to crawl the individual files

    Args:
        root (string): Root directory of dataset.
    """
    
    def __init__(self,
                root: str):
        
        self.root = root
