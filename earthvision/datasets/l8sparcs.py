"""Landsat 8 SPARCS Cloud Dataset."""
from PIL import Image
import os
import shutil
import posixpath
import numpy as np
import pandas as pd
import glob

from typing import Any, Callable, Optional, Tuple
from .utils import _urlretrieve, _load_img
from .vision import VisionDataset


class L8SPARCS(VisionDataset):
    """Landsat 8 SPARCS Cloud.
    
    <https://www.usgs.gov/core-science-systems/nli/landsat/spatial-procedures-automated-removal-cloud-and-shadow-sparcs>
    
    Download: <https://landsat.usgs.gov/cloud-validation/sparcs/l8cloudmasks.zip>

    Args:
        root (string): Root directory of dataset.
        transform (callable, optional): A function/transform that  takes in an PIL image and
            returns a transformed version. E.g, transforms.RandomCrop
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    mirrors = "https://landsat.usgs.gov/cloud-validation/sparcs/"
    resources = "l8cloudmasks.zip"

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:

        super(L8SPARCS, self).__init__(root, transform=transform, target_transform=target_transform)

        self.root = root
        self.data_mode = "sending"

        if download and self._check_exists():
            print("file already exists.")

        if download and not self._check_exists():
            self.download()
            self.extract_file()

        self.img_labels = self.get_path_and_label()

    def _check_exists(self) -> None:
        self.data_path = os.path.join(self.root, self.data_mode)
        return os.path.exists(self.data_path)

    def download(self) -> None:
        """Download file"""
        file_url = posixpath.join(self.mirrors, self.resources)
        _urlretrieve(file_url, os.path.join(self.root, self.resources))

    def extract_file(self) -> None:
        """Extract the .zip file"""
        shutil.unpack_archive(os.path.join(self.root, self.resources), self.root)
        os.remove(os.path.join(self.root, self.resources))

    def get_path_and_label(self):
        """Get the path of the images and labels (masks) in a dataframe"""
        image_path, label = [], []

        for image in glob.glob(os.path.join(self.root, self.data_mode, "*_photo.png")):
            image_path.append(image)

        for mask in glob.glob(os.path.join(self.root, self.data_mode, "*_mask.png")):
            label.append(mask)

        df = pd.DataFrame({"image": sorted(image_path), "label": sorted(label)})

        return df

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """
        Args:
            idx (int): Index
        Returns:
            tuple: (img, mask)
        """
        img_path = self.img_labels.iloc[idx, 0]
        mask_path = self.img_labels.iloc[idx, 1]

        img = np.array(_load_img(img_path))
        mask = np.array(_load_img(mask_path))

        if self.transform is not None:
            img = Image.fromarray(img)
            img = self.transform(img)

        if self.target_transform is not None:
            mask = Image.fromarray(mask)
            mask = self.target_transform(mask)
        return img, mask

    def __len__(self) -> int:
        return len(self.img_labels)
