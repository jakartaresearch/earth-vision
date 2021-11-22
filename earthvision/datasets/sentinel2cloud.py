"""Sentinel-2 Cloud Mask Catalogue Dataset."""
from PIL import Image
import os
import shutil
import posixpath
import pandas as pd
import glob

from typing import Any, Callable, Optional, Tuple
from .vision import VisionDataset
from .utils import _urlretrieve, _load_npy


class Sentinel2Cloud(VisionDataset):
    """Sentinel-2 Cloud Mask Catalogue dataset.
    
    classification_tags: <https://zenodo.org/record/4172871/files/classification_tags.csv?download=1>
    subscenes: <https://zenodo.org/record/4172871/files/subscenes.zip?download=1>
    masks: <https://zenodo.org/record/4172871/files/masks.zip?download=1>

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

    mirrors = "https://zenodo.org/record/4172871/files/"
    resources = "subscenes.zip"
    mask_resources = "masks.zip"

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:

        super(Sentinel2Cloud, self).__init__(
            root, transform=transform, target_transform=target_transform
        )

        self.root = root

        if not os.path.exists(self.root):
            os.makedirs(self.root)

        if download and self._check_exists():
            print("file already exists.")

        if download and not self._check_exists():
            self.download()
            self.extract_file()

        self.img_labels = self.get_image_path_and_mask_path()

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """
        Args:
            idx (int): Index
        Returns:
            tuple: (img, mask)
        """
        img_path = self.img_labels.iloc[idx, 0]
        mask_path = self.img_labels.iloc[idx, 1]

        img = _load_npy(img_path)
        mask = _load_npy(mask_path)

        if self.transform is not None:
            img = Image.fromarray(img)
            img = self.transform(img)

        if self.target_transform is not None:
            mask = Image.fromarray(mask)
            mask = self.target_transform(mask)
        return img, mask

    def __len__(self) -> int:
        """Return the len of the image labels"""
        return len(self.img_labels)

    def get_image_path_and_mask_path(self):
        """Return dataframe type consist of image path and mask path."""

        img_path = os.path.join(self.root, "sentinel2cloud", "subscenes")
        msk_path = os.path.join(self.root, "sentinel2cloud", "masks")

        images_path = glob.glob(os.path.join(img_path, "*.npy"))
        images_path.sort()
        masks_path = glob.glob(os.path.join(msk_path, "*.npy"))
        masks_path.sort()

        df = pd.DataFrame({"image": images_path, "mask": masks_path})
        return df

    def download(self) -> None:
        """download and extract file."""
        file_url = posixpath.join(self.mirrors, self.resources)
        _urlretrieve(file_url, os.path.join(self.root, self.resources))

        mask_file_url = posixpath.join(self.mirrors, self.mask_resources)
        _urlretrieve(mask_file_url, os.path.join(self.root, self.mask_resources))

    def _check_exists(self):
        """Check file has been download or not"""
        self.data_path = os.path.join(self.root, "sentinel2cloud")

        return os.path.exists(os.path.join(self.data_path, "subscenes")) and os.path.exists(
            os.path.join(self.data_path, "masks")
        )

    def extract_file(self):
        """Extract file from compressed."""

        os.makedirs(os.path.join(self.root, "sentinel2cloud"))

        shutil.unpack_archive(
            os.path.join(self.root, self.resources), os.path.join(self.root, "sentinel2cloud")
        )
        os.remove(os.path.join(self.root, self.resources))

        shutil.unpack_archive(
            os.path.join(self.root, self.mask_resources), os.path.join(self.root, "sentinel2cloud")
        )
        os.remove(os.path.join(self.root, self.mask_resources))
