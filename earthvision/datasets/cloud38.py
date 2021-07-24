import os
import shutil
import posixpath
from sys import path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Resize
from .utils import _urlretrieve, _load_img
from PIL import Image
from pathlib import Path
import glob
import cv2


class Cloud38(Dataset):
    """
    """

    mirrors = "https://storage.googleapis.com/ossjr/"
    resources = "38cloud.zip"

    def __init__(self,
                 root: str):

        self.root = root
        self.data_path = os.path.join(self.root, "38cloud")
        self.base_path = Path(os.path.join(self.data_path, os.path.join("38cloud", "38-Cloud_training")))

        if not os.path.exists(self.root):
            os.makedirs(self.root)

        if not self._check_exists():
            self.download()
            self.extract_file()

        if not (self.base_path/'train_rgb').exists():
            (self.base_path/'train_rgb').mkdir()
        
        if not (self.base_path/'labels').exists():
            (self.base_path/'labels').mkdir()

        for red_patch in (self.base_path/'train_red').iterdir():
            self.create_rgb_pil(red_patch)

        for gt_patch in (self.base_path/'train_gt').iterdir():
            self.convert_tif_png(gt_patch, self.base_path/'labels')
        
        print("Done.")


    def create_rgb_pil(self, red_filename: Path):
        """
        """
        self.red_filename = str(red_filename)
        green_fn = self.red_filename.replace('red', 'green')
        blue_fn = self.red_filename.replace('red', 'blue')
        rgb_fn = self.red_filename.replace('red', 'rgb').replace('.TIF', '.png')

        array_red = np.array(Image.open(self.red_filename))
        array_green = np.array(Image.open(green_fn))
        array_blue = np.array(Image.open(blue_fn))

        array_rgb = np.stack([array_red, array_green, array_blue], axis=2)
        array_rgb = array_rgb / np.iinfo(array_rgb.dtype).max

        rgb = Image.fromarray((256*array_rgb).astype(np.uint8), 'RGB')
        rgb.save(rgb_fn)
        return rgb

    def convert_tif_png(self, tif_file: Path, out_folder:Path):
        """
        """
        self.tif_file = tif_file
        self.out_folder = out_folder
        array_tif = np.array(Image.open(self.tif_file))
        im = Image.fromarray(np.where(array_tif==255, 1, 0))
        im.save(self.out_folder/self.tif_file.with_suffix('.png').name)
        return im
      
    def __len__(self):
        return len(self.img_labels)

    def __iter__(self):
        for index in range(self.__len__()):
            yield self.__getitem__(index)

    def download(self):
        """download and extract file.
        """
        file_url = posixpath.join(self.mirrors, self.resources)
        _urlretrieve(file_url, os.path.join(self.root, self.resources))

    def _check_exists(self):
        """Check file has been download or not
        """

        return os.path.exists(os.path.join(self.data_path, os.path.join("38cloud", "38-Cloud_95-Cloud_Test_Metadata_Files"))) and \
            os.path.exists(os.path.join(self.data_path, os.path.join("38cloud", "38-Cloud_test"))) and \
            os.path.exists(os.path.join(self.data_path, os.path.join("38cloud", "38-Cloud_training"))) and \
            os.path.exists(os.path.join(self.data_path, os.path.join("38cloud", "38-Cloud_Training_Metadata_Files")))

    def extract_file(self):
        """
        """
        print("Extracting...")
        shutil.unpack_archive(os.path.join(self.root, self.resources), os.path.join(self.root, "38cloud"))
        os.remove(os.path.join(self.root, self.resources))