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

class SpaceNet7():
    """SpaceNet7 . 
    SN7: Multi-Temporal Urban Development Challenge
    <https://spacenet.ai/sn7-challenge/>

    Args:
        root (string): Root directory of dataset.
    """
    

    def __init__(self,
                root: str):
                
        self.root = root
    
    def download(self):
        pass

    def extract_file(self):
        pass

