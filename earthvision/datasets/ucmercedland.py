"""UC Merced Land Use Dataset."""
from PIL import Image
import os
import shutil
import posixpath
import numpy as np
import pandas as pd

from typing import Any, Callable, Optional, Tuple
from torchvision.transforms import Resize, ToTensor, Compose
from .vision import VisionDataset
from .utils import _urlretrieve, _load_img


class UCMercedLand(VisionDataset):
    """UC Merced Land Use Dataset.

    <http://weegee.vision.ucmerced.edu/datasets/UCMerced_LandUse.zip>

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

    mirrors = "http://weegee.vision.ucmerced.edu/datasets/"
    resources = "UCMerced_LandUse.zip"
    classes = {
        "agricultural": 0,
        "airplane": 1,
        "baseballdiamond": 2,
        "beach": 3,
        "buildings": 4,
        "chaparral": 5,
        "denseresidential": 6,
        "forest": 7,
        "freeway": 8,
        "golfcourse": 9,
        "harbor": 10,
        "intersection": 11,
        "mediumresidential": 12,
        "mobilehomepark": 13,
        "overpass": 14,
        "parkinglot": 15,
        "river": 16,
        "runway": 17,
        "sparseresidential": 18,
        "storagetanks": 19,
        "tenniscourt": 20,
    }

    def __init__(
        self,
        root: str,
        transform=Compose([Resize((256, 256)), ToTensor()]),
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:

        super(UCMercedLand, self).__init__(
            root, transform=transform, target_transform=target_transform
        )

        self.root = root
        self.data_mode = "Images"

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
        image_path = []
        label = []
        for cat, enc in self.classes.items():
            cat_path = os.path.join(self.root, "UCMerced_LandUse", self.data_mode, cat)
            cat_image = [os.path.join(cat_path, path) for path in os.listdir(cat_path)]
            cat_label = [enc] * len(cat_image)
            image_path += cat_image
            label += cat_label
        df = pd.DataFrame({"image": image_path, "label": label})

        return df

    def _check_exists(self):
        self.data_path = os.path.join(self.root, "UCMerced_LandUse", "Images")
        self.dir_classes = list(self.classes.keys())
        return all([os.path.exists(os.path.join(self.data_path, i)) for i in self.dir_classes])

    def download(self) -> None:
        """download and extract file."""
        file_url = posixpath.join(self.mirrors, self.resources)
        _urlretrieve(file_url, os.path.join(self.root, self.resources))

    def extract_file(self) -> None:
        """Extract file from compressed."""
        shutil.unpack_archive(os.path.join(self.root, self.resources), self.root)
        os.remove(os.path.join(self.root, self.resources))
