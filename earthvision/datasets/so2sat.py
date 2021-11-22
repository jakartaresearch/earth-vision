"""So2Sat Dataset to Predict Local Climate Zone (LCZ)."""
from PIL import Image
import os
import posixpath
import numpy as np
import h5py

from typing import Any, Callable, Optional, Tuple
from .utils import _urlretrieve
from .vision import VisionDataset


class So2Sat(VisionDataset):
    """So2Sat Dataset to Predict Local Climate Zone (LCZ): 
    
    <https://mediatum.ub.tum.de/1454690>

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

    mirrors = "https://dataserv.ub.tum.de/s/m1454690/download?path=/&files="
    resources = ["training.h5", "validation.h5"]

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = True,
    ) -> None:

        super(So2Sat, self).__init__(root, transform=transform, target_transform=target_transform)

        self.root = root
        self.data_mode = "training" if train else "validation"

        if download and self._check_exists():
            print("file already exists.")

        if download and not self._check_exists():
            self.download()

        self.img_labels = self.get_path_and_label()

    def __len__(self) -> int:
        return len(self.img_labels)

    def __getitem__(self, idx: int) -> Tuple[Any, Any, Any]:
        """
        Args:
            idx (int): Index
        Returns:
            tuple: (sen1, sen2, label)
        """
        sen1 = self.img_labels["sen1"][idx]
        sen2 = self.img_labels["sen2"][idx]
        label = self.img_labels["label"][idx]

        if self.transform is not None:
            sen1 = Image.fromarray(sen1)
            sen1 = self.transform(sen1)

            sen2 = Image.fromarray(sen2)
            sen2 = self.transform(sen2)

        if self.target_transform is not None:
            label = Image.fromarray(label)
            label = self.target_transform(label)

        return (sen1, sen2, label)

    def get_path_and_label(self):
        """Return dataframe type consist of image path and corresponding label."""
        file = h5py.File(os.path.join(self.root, f"{self.data_mode}.h5"), "r")

        sen1 = np.array(file["sen1"])
        sen2 = np.array(file["sen2"])
        label = np.array(file["label"])

        return {"sen1": sen1, "sen2": sen2, "label": label}

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.resources[0])) and os.path.exists(
            os.path.join(self.root, self.resources[1])
        )

    def download(self):
        """Download and extract file."""
        if not os.path.exists(self.root):
            os.makedirs(self.root)

        for resource in self.resources:
            file_url = posixpath.join(self.mirrors, resource)
            _urlretrieve(file_url, os.path.join(self.root, resource))
