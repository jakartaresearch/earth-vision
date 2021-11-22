"""L8 Biome Cloud Cover Dataset."""
from PIL import Image
import os
import shutil
import pandas as pd
import glob
import requests

from bs4 import BeautifulSoup
from typing import Any, Callable, Optional, Tuple
from .vision import VisionDataset
from .utils import _urlretrieve, _load_img_hdr, _load_stack_img


class L8Biome(VisionDataset):
    """L8 Biome Cloud Cover.

    Download page https://landsat.usgs.gov/landsat-8-cloud-cover-assessment-validation-data

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

    mirrors = "https://landsat.usgs.gov/landsat-8-cloud-cover-assessment-validation-data"

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:

        super(L8Biome, self).__init__(root, transform=transform, target_transform=target_transform)

        self.root = root
        self.download_urls = self.get_download_url()
        self.data_modes = [url.split("/")[-1] for url in self.download_urls]

        if download and self._check_exists():
            print("file already exists.")

        if download and not self._check_exists():
            self.download()
            self.extract_file()

        self.img_labels = self.get_path_and_label()

    def get_download_url(self):
        """Get the urls to download the files."""
        page = requests.get(self.mirrors)
        soup = BeautifulSoup(page.content, "html.parser")

        urls = [url.get("href") for url in soup.find_all("a")]

        download_urls = list(filter(lambda url: url.endswith(".tar.gz") if url else None, urls))
        return download_urls

    def download(self):
        """Download file"""
        for resource in self.download_urls:
            filename = resource.split("/")[-1]
            _urlretrieve(resource, os.path.join(self.root, filename))

    def extract_file(self):
        """Extract the .zip file"""
        for resource in self.data_modes:
            shutil.unpack_archive(os.path.join(self.root, resource), self.root)
            os.remove(os.path.join(self.root, resource))

    def _check_exists(self):
        is_exists = []
        if not os.path.isdir(self.root):
            os.mkdir(self.root)

        for data_mode in self.data_modes:
            data_mode = data_mode.replace(".tar.gz", "")
            data_path = os.path.join(self.root, "BC", data_mode)
            is_exists.append(os.path.exists(data_path))

        return all(is_exists)

    def get_path_and_label(self):
        """Get the path of the images and labels (masks) in a dataframe"""
        image_directory, label = [], []

        for data_mode in self.data_modes:
            data_mode = data_mode.replace(".tar.gz", "")
            image_dir = os.path.join(self.root, "BC", data_mode)

            image_directory.append(image_dir)
            label.extend(glob.glob(os.path.join(self.root, "BC", data_mode, "*mask.hdr")))

        df = pd.DataFrame({"image": image_directory, "label": label})
        return df

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """
        Args:
            idx (int): Index
        Returns:
            tuple: (img, mask)
        """
        img_directory = self.img_labels.iloc[idx, 0]
        mask_path = self.img_labels.iloc[idx, 1]

        ls_stack_path = []
        for idx in range(1, 12):
            observation = img_directory.split("/")[-1]
            name_file = f"{img_directory}/{observation}_B{idx}.TIF"
            ls_stack_path.append(name_file)

        img = _load_stack_img(ls_stack_path)
        mask = _load_img_hdr(mask_path)

        if self.transform is not None:
            img = Image.fromarray(img)
            img = self.transform(img)

        if self.target_transform is not None:
            mask = Image.fromarray(mask)
            mask = self.target_transform(mask)

        return img, mask

    def __len__(self) -> int:
        return len(self.img_labels)
