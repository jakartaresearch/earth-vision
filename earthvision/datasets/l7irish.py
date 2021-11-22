"""Landsat 7 Irish Cloud Dataset."""
from PIL import Image
import os
import shutil
import posixpath
import numpy as np
import pandas as pd
import glob
import requests

from typing import Any, Callable, Optional, Tuple
from .vision import VisionDataset
from .utils import _urlretrieve, _load_img
from bs4 import BeautifulSoup


class L7Irish(VisionDataset):
    """Landsat 7 Irish Cloud.

    <https://landsat.usgs.gov/landsat-7-cloud-cover-assessment-validation-data>

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

    mirrors = "http://landsat.usgs.gov/cloud-validation/cca_irish_2015/"

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:

        super(L7Irish, self).__init__(root, transform=transform, target_transform=target_transform)

        self.root = root
        self.download_urls = self.get_download_url()
        self.resources = [url.split("/")[-1] for url in self.download_urls]
        self.data_modes = [filename.split(".tar.gz")[0] for filename in self.resources]

        if download and self._check_exists():
            print("file already exists.")

        if download and not self._check_exists():
            self.download()
            self.extract_file()

        self.img_labels = self.get_path_and_label()

    def get_download_url(self):
        """Get the urls to download the files."""
        page = requests.get(
            "https://landsat.usgs.gov/landsat-7-cloud-cover-assessment-validation-data"
        )
        soup = BeautifulSoup(page.content, "html.parser")

        urls = [url.get("href") for url in soup.find_all("a")]
        urls = list(filter(None, urls))

        download_urls = filter(lambda url: url.endswith(".gz"), urls)
        return download_urls

    def download(self):
        """Download file"""
        for resource in self.resources:
            file_url = posixpath.join(self.mirrors, resource)
            _urlretrieve(file_url, os.path.join(self.root, resource))

    def extract_file(self):
        """Extract the .zip file"""
        for resource in self.resources:
            shutil.unpack_archive(os.path.join(self.root, resource), self.root)
            os.remove(os.path.join(self.root, resource))

    def _check_exists(self):
        is_exists = []
        if not os.path.isdir(self.root):
            os.mkdir(self.root)

        for data_mode in self.data_modes:
            data_path = os.path.join(self.root, data_mode)
            is_exists.append(os.path.exists(data_path))

        return all(is_exists)

    def get_path_and_label(self):
        """Get the path of the images and labels (masks) in a dataframe"""
        image_path, label = [], []

        for data_mode in self.data_modes:
            for image in glob.glob(os.path.join(self.root, data_mode, "L7*.TIF")):
                image_path.append(image)

                label.extend(glob.glob(os.path.join(self.root, data_mode, "*mask*")))

        df = pd.DataFrame({"image": image_path, "label": label})
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
