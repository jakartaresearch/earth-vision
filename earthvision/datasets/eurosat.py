"""EuroSat Land Cover Categories Dataset."""
from PIL import Image
import os
import shutil
import posixpath
import numpy as np
import pandas as pd

from typing import Any, Callable, Optional, Tuple
from .utils import _urlretrieve, _load_img
from .vision import VisionDataset
from torchvision.transforms import Resize, ToTensor, Compose


class EuroSat(VisionDataset):
    """EuroSat Land Cover Categories.

    <http://madm.dfki.de/files/sentinel>

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

    mirrors = "http://madm.dfki.de/files/sentinel"
    resources = "EuroSAT.zip"
    classes = {
        "AnnualCrop": 0,
        "Forest": 1,
        "HerbaceousVegetation": 2,
        "Highway": 3,
        "Industrial": 4,
        "Pasture": 5,
        "PermanentCrop": 6,
        "Residential": 7,
        "River": 8,
        "SeaLake": 9,
    }

    def __init__(
        self,
        root: str,
        transform=Compose([Resize((64, 64)), ToTensor()]),
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:

        super(EuroSat, self).__init__(root, transform=transform, target_transform=target_transform)

        self.root = root
        self.data_mode = "2750"

        if download and self._check_exists():
            print("file already exists.")

        if download and not self._check_exists():
            self.download()
            self.extract_file()

        self.img_labels = self.get_path_and_label()

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """
        Args:
            idx (int): Index
        Returns:
            tuple: (img, target) where target is index of the target class.
        """
        img_path = self.img_labels.iloc[idx, 0]
        img = np.array(_load_img(img_path))
        target = self.img_labels.iloc[idx, 1]

        if self.transform is not None:
            img = Image.fromarray(img)
            img = self.transform(img)

        if self.target_transform is not None:
            target = Image.fromarray(target)
            target = self.target_transform(target)
        return img, target

    def __len__(self) -> int:
        return len(self.img_labels)

    def _check_exists(self) -> None:
        self.data_path = os.path.join(self.root, self.data_mode)
        self.dir_classes = list(self.classes.keys())

        return all([os.path.exists(os.path.join(self.data_path, i)) for i in self.dir_classes])

    def download(self) -> None:
        """Download file"""
        file_url = posixpath.join(self.mirrors, self.resources)
        _urlretrieve(file_url, os.path.join(self.root, self.resources))

    def extract_file(self) -> None:
        """Extract the .zip file"""
        shutil.unpack_archive(os.path.join(self.root, self.resources), self.root)
        os.remove(os.path.join(self.root, self.resources))

    def get_path_and_label(self):
        """Return dataframe type consist of image path and corresponding label."""
        image_path = []
        label = []
        for cat, enc in self.classes.items():
            cat_path = os.path.join(self.root, self.data_mode, cat)
            cat_image = [os.path.join(cat_path, path) for path in os.listdir(cat_path)]
            cat_label = [enc] * len(cat_image)
            image_path += cat_image
            label += cat_label
        df = pd.DataFrame({"image": image_path, "label": label})

        return df
