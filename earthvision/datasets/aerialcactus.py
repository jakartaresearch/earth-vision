"""Aerial Cactus Dataset from Kaggle."""
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


class AerialCactus(VisionDataset):
    """Aerial Cactus Dataset.
    
    <https://www.kaggle.com/c/aerial-cactus-identification>

    Args:
        root (string): Root directory of dataset.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from validation set.
        transform (callable, optional): A function/transform that  takes in an PIL image and
            returns a transformed version. E.g, transforms.RandomCrop
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    mirrors = "https://storage.googleapis.com/ossjr"
    resources = "cactus-aerial-photos.zip"

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform=Compose([Resize((32, 32)), ToTensor()]),
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:

        super(AerialCactus, self).__init__(
            root, transform=transform, target_transform=target_transform
        )

        self.root = root
        self.data_mode = "training_set" if train else "validation_set"

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
        classes = {"cactus": 1, "no_cactus": 0}
        image_path, label = [], []

        for cat, enc in classes.items():
            cat_path = os.path.join(
                self.root, "cactus-aerial-photos", self.data_mode, self.data_mode, cat
            )
            cat_image = [os.path.join(cat_path, path) for path in os.listdir(cat_path)]
            cat_label = [enc] * len(cat_image)
            image_path += cat_image
            label += cat_label
        df = pd.DataFrame({"image": image_path, "label": label})

        return df

    def _check_exists(self):
        self.train_path = os.path.join(
            self.root, "cactus-aerial-photos", "training_set", "training_set"
        )
        self.valid_path = os.path.join(
            self.root, "cactus-aerial-photos", "validation_set", "validation_set"
        )

        folder_status = []
        for path in [self.train_path, self.valid_path]:
            for target in ["cactus", "no_cactus"]:
                folder_status.append(os.path.exists(os.path.join(path, target)))

        return all(folder_status)

    def download(self) -> None:
        """Download and extract file."""
        os.makedirs(self.root, exist_ok=True)

        file_url = posixpath.join(self.mirrors, self.resources)
        _urlretrieve(file_url, os.path.join(self.root, self.resources))

    def extract_file(self) -> None:
        """Extract file from compressed."""
        path_destination = os.path.join(self.root, "cactus-aerial-photos")
        shutil.unpack_archive(os.path.join(self.root, self.resources), path_destination)
        os.remove(os.path.join(self.root, self.resources))
