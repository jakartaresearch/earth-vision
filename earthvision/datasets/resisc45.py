"""RESISC45 Dataset."""
from PIL import Image
import os
import posixpath
import shutil
import numpy as np
import pandas as pd

from typing import Any, Callable, Optional, Tuple
from .vision import VisionDataset
from earthvision.constants.RESISC45.config import CLASS_ENC, CLASS_DEC
from earthvision.datasets.utils import _urlretrieve, _load_img


class RESISC45(VisionDataset):
    """RESISC45 Dataset.

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

    mirrors = "https://storage.googleapis.com/ossjr"
    resources = "NWPU-RESISC45.zip"

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:

        super(RESISC45, self).__init__(root, transform=transform, target_transform=target_transform)

        self.root = root
        self.class_enc = CLASS_ENC
        self.class_dec = CLASS_DEC

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

    def get_path_and_label(self):
        """Return dataframe type consist of image path and corresponding label."""
        DATA_SIZE = 700
        category = os.listdir(os.path.join(self.root, "NWPU-RESISC45"))
        image_path = []
        label = []
        for cat in category:
            cat_enc = self.class_enc[cat]
            label += [cat_enc] * DATA_SIZE
            for num in range(1, DATA_SIZE + 1):
                filename = cat + "_" + str(num).zfill(3) + ".jpg"
                image_path += [os.path.join(self.root, "NWPU-RESISC45", cat, filename)]
        df = pd.DataFrame({"image": image_path, "label": label})

        return df

    def _check_exists(self):
        is_exists = os.path.exists(os.path.join(self.root, "NWPU-RESISC45"))
        return is_exists

    def download(self) -> None:
        """Download and extract file."""
        file_url = posixpath.join(self.mirrors, self.resources)
        _urlretrieve(file_url, os.path.join(self.root, self.resources))

    def extract_file(self) -> None:
        """Extract file from compressed."""
        shutil.unpack_archive(os.path.join(self.root, self.resources), f"{self.root}")
        os.remove(os.path.join(self.root, self.resources))
