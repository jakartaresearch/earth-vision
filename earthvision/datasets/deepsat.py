"""Deepsat Dataset - Scene Classification."""
from PIL import Image
import os
import scipy.io as sio
import gdown
import sys

from typing import Any, Callable, Optional, Tuple
from .vision import VisionDataset


class DeepSat(VisionDataset):
    """DeepSat Dataset.

    Args:
        root (string): Root directory of dataset.
        dataset_type (string, optional): Choose dataset type ['SAT-4', 'SAT-6'].
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image and
            returns a transformed version. E.g, transforms.RandomCrop
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    resources = {
        "SAT-4_and_SAT-6_datasets": "https://drive.google.com/uc?id=0B0Fef71_vt3PUkZ4YVZ5WWNvZWs&export=download"
    }
    dataset_types = ["SAT-4", "SAT-6"]

    def __init__(
        self,
        root: str,
        dataset_type="SAT-4",
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:

        super(DeepSat, self).__init__(root, transform=transform, target_transform=target_transform)

        self.root = root
        self.dataset_type = dataset_type
        self.train = train
        self.folder_pth = os.path.join(self.root, list(self.resources.keys())[0])
        self.filename = list(self.resources.keys())[0] + ".tar.gz"

        if download and self._check_exists():
            print("file already exists.")

        if download and not self._check_exists():
            self.download()

        dataset = self.load_dataset()
        self.choose_data_mode(dataset)

    def download(self) -> None:
        """Download dataset and extract it"""

        self.root = os.path.expanduser(self.root)
        print("Download dataset...")

        gdown.download(
            self.resources["SAT-4_and_SAT-6_datasets"],
            os.path.join(self.root, self.filename),
            quiet=False,
        )

        if os.path.exists(self.folder_pth):
            print(f"file {self.folder_pth} already exists")
        else:
            os.mkdir(self.folder_pth)
            print(f"Extracting file {self.filename}")
            os.system(f"tar -xvf {os.path.join(self.root, self.filename)} -C {self.folder_pth}")
            os.system(f"mv {self.folder_pth} {self.root}")
            print("Extracting file success !")

    def _check_exists(self) -> bool:
        if self.dataset_type not in self.dataset_types:
            print(f"Unknown dataset {self.dataset_type}")
            print(f"Available dataset : {self.dataset_types}")
            sys.exit(0)

        if os.path.exists(self.filename):
            return True
        else:
            return False

    def load_dataset(self):
        filename = {"SAT-4": "sat-4-full.mat", "SAT-6": "sat-6-full.mat"}
        dataset = sio.loadmat(os.path.join(self.folder_pth, filename[self.dataset_type]))
        return dataset

    def choose_data_mode(self, dataset):
        if self.train:
            x_type, y_type = "train_x", "train_y"
        else:
            x_type, y_type = "test_x", "test_y"

        self.x, self.y = dataset[x_type], dataset[y_type]
        self.annot = dataset["annotations"]

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """
        Args:
            idx (int): Index
        Returns:
            tuple: (img, target) where target is index of the target class.
        """
        img = self.x[:, :, :, idx]
        target = self.y[:, idx]

        if self.transform is not None:
            img = Image.fromarray(img)
            img = self.transform(img)

        if self.target_transform is not None:
            target = Image.fromarray(target)
            target = self.target_transform(target)
        return img, target

    def __len__(self) -> int:
        return self.x.shape[3]
