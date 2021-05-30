import os
import shutil
import posixpath
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Resize
from .utils import _urlretrieve, _load_img


class UCMercedLand(Dataset):
    """UC Merced Land Use Dataset.
    <http://weegee.vision.ucmerced.edu/datasets/UCMerced_LandUse.zip>
    """

    mirrors = "http://weegee.vision.ucmerced.edu/datasets/"
    resources = "UCMerced_LandUse.zip"

    def __init__(self,
                 root: str,
                 data_mode: str = 'Images',
                 transform=Resize((256, 256)),
                 target_transform=None):

        self.root = root
        self.data_mode = data_mode
        self.transform = transform
        self.target_transform = target_transform

        if not self._check_exists():
            self.download()
            self.extract_file()

        self.img_labels = self.get_path_and_label()

    def __getitem__(self, idx):
        img_path = self.img_labels.iloc[idx, 0]
        label = self.img_labels.iloc[idx, 1]
        image = _load_img(img_path)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        image = np.array(image)
        image = torch.from_numpy(image)
        sample = (image, label)

        return sample

    def __len__(self):
        return len(self.img_labels)
    
    def __iter__(self):
        for index in range(self.__len__()):
            yield self.__getitem__(index)
        
    def get_path_and_label(self):
        """Return dataframe type consist of image path and corresponding label."""
        classes = {'agricultural': 0, \
                    'airplane': 1, \
                    'baseballdiamond': 2, \
                    'beach': 3, \
                    'buildings': 4, \
                    'chaparral': 5, \
                    'denseresidential': 6, \
                    'forest': 7, \
                    'freeway': 8, \
                    'golfcourse': 9, \
                    'harbor': 10, \
                    'intersection': 11, \
                    'mediumresidential': 12, \
                    'mobilehomepark': 13, \
                    'overpass': 14, \
                    'parkinglot': 15, \
                    'river': 16, \
                    'runway': 17, \
                    'sparseresidential': 18, \
                    'storagetanks': 19, \
                    'tenniscourt': 20}
        image_path = []
        label = []
        for cat, enc in classes.items():
            cat_path = os.path.join(
                self.root, 'UCMerced_LandUse', 'UCMerced_LandUse', self.data_mode, cat)
            cat_image = [os.path.join(cat_path, path)
                         for path in os.listdir(cat_path)]
            cat_label = [enc] * len(cat_image)
            image_path += cat_image
            label += cat_label
        df = pd.DataFrame({'image': image_path, 'label': label})

        return df

    def _check_exists(self):
        self.data_path = os.path.join(
            self.root, "UCMerced_LandUse", "UCMerced_LandUse", "Images")

        return os.path.exists(os.path.join(self.data_path, "agricultural")) and \
            os.path.exists(os.path.join(self.data_path, "airplane")) and \
            os.path.exists(os.path.join(self.data_path, "baseballdiamond")) and \
            os.path.exists(os.path.join(self.data_path, "beach")) and \
            os.path.exists(os.path.join(self.data_path, "buildings")) and \
            os.path.exists(os.path.join(self.data_path, "chaparral")) and \
            os.path.exists(os.path.join(self.data_path, "denseresidential")) and \
            os.path.exists(os.path.join(self.data_path, "forest")) and \
            os.path.exists(os.path.join(self.data_path, "freeway")) and \
            os.path.exists(os.path.join(self.data_path, "golfcourse")) and \
            os.path.exists(os.path.join(self.data_path, "harbor")) and \
            os.path.exists(os.path.join(self.data_path, "intersection")) and \
            os.path.exists(os.path.join(self.data_path, "mediumresidential")) and \
            os.path.exists(os.path.join(self.data_path, "mobilehomepark")) and \
            os.path.exists(os.path.join(self.data_path, "overpass")) and \
            os.path.exists(os.path.join(self.data_path, "parkinglot")) and \
            os.path.exists(os.path.join(self.data_path, "river")) and \
            os.path.exists(os.path.join(self.data_path, "runway")) and \
            os.path.exists(os.path.join(self.data_path, "sparseresidential")) and \
            os.path.exists(os.path.join(self.data_path, "storagetanks")) and \
            os.path.exists(os.path.join(self.data_path, "tenniscourt"))

    def download(self):
        """download and extract file."""
        file_url = posixpath.join(self.mirrors, self.resources)
        _urlretrieve(file_url, os.path.join(self.root, self.resources))
    
    def extract_file(self):
        """Extract file from compressed."""
#         path_destination = os.path.join(
#             self.root, self.resources.replace(".zip", ""))
#         os.makedirs(path_destination, exist_ok=True)
        shutil.unpack_archive(os.path.join(self.root, self.resources), self.root)
        os.remove(os.path.join(self.root, self.resources))
