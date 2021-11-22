"""Cars Overhead with Context Dataset."""
from PIL import Image
import os
import shutil
import posixpath
import tarfile
import numpy as np
import pandas as pd

from typing import Any, Callable, Optional, Tuple
from .vision import VisionDataset
from .utils import _urlretrieve, _load_img
from ..constants.COWC.config import file_mapping_counting, file_mapping_detection


class COWC(VisionDataset):
    """Cars Overhead with Context.
    
    https://gdo152.llnl.gov/cowc/

    Args:
        root (string): Root directory of dataset.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        task_mode (string): There is 2 task mode i.e. 'counting' and 'detection'. Default value is 'counting'.
        transform (callable, optional): A function/transform that  takes in an PIL image and
            returns a transformed version. E.g, transforms.RandomCrop
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    mirrors = "https://gdo152.llnl.gov/cowc/download"
    resources = "cowc-everything.txz"

    def __init__(
        self,
        root: str,
        train: bool = True,
        task_mode: str = "counting",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:

        super(COWC, self).__init__(root, transform=transform, target_transform=target_transform)

        self.root = root
        self.train = train
        self.task_mode = task_mode

        if download and self._check_exists():
            print("file already exists.")

        if download and not self._check_exists():
            self.download()
            self.extract_file()

        if self.task_mode == "counting":
            self.task_path = os.path.join(self.root, "cowc/datasets/patch_sets/counting")
            self.file_mapping = file_mapping_counting
        elif self.task_mode == "detection":
            self.task_path = os.path.join(self.root, "cowc/datasets/patch_sets/detection")
            self.file_mapping = file_mapping_detection
        else:
            raise ValueError("task_mode not recognized.")

        for filename, compressed in self.file_mapping.items():
            if not self._check_exists_subfile(filename):
                self.extract_subfile(filename, compressed)

        self.img_labels = self.get_path_and_label()

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """
        Args:
            idx (int): Index
        Returns:
            tuple: (img, target) where target is index of the target class.
        """
        img_path = self.img_labels.iloc[idx, 0]
        target = self.img_labels.iloc[idx, 1]
        folder = img_path.split("/", 1)[0]
        img_path = os.path.join(self.task_path, folder, img_path)
        img = np.array(_load_img(img_path))

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
        """Return dataframe type consist of image path
        and corresponding label."""

        if self.task_mode == "counting":
            if self.train:
                label_name = "COWC_train_list_64_class.txt.bz2"
            else:
                label_name = "COWC_test_list_64_class.txt.bz2"

        elif self.task_mode == "detection":
            if self.train:
                label_name = "COWC_train_list_detection.txt.bz2"
            else:
                label_name = "COWC_test_list_detection.txt.bz2"

        else:
            raise ValueError("task_mode not recognized.")

        label_path = os.path.join(self.task_path, label_name)
        df = pd.read_csv(label_path, sep=" ", header=None)

        return df

    def _check_exists_subfile(self, filename):
        path_to_check = os.path.join(self.task_path, filename)
        return os.path.exists(path_to_check)

    def extract_subfile(self, filename, compressed):
        comp_path = os.path.join(self.task_path, compressed)
        file_path = os.path.join(self.task_path, filename)
        tar = tarfile.open(comp_path)
        tar.extractall(file_path)
        tar.close()

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, "cowc"))

    def download(self) -> None:
        """download file."""
        file_url = posixpath.join(self.mirrors, self.resources)
        _urlretrieve(file_url, os.path.join(self.root, self.resources))

    def extract_file(self) -> None:
        """Extract file from compressed."""
        shutil.unpack_archive(os.path.join(self.root, self.resources), self.root)
        os.remove(os.path.join(self.root, self.resources))
