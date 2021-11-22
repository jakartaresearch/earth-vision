"""38-Cloud: A Cloud Segmentation Dataset."""
# Reference https://github.com/cordmaur/38Cloud-Medium
from PIL import Image
from pathlib import Path
import os
import shutil
import numpy as np
import pandas as pd

from typing import Any, Callable, Optional, Tuple
from .utils import _urlretrieve, _load_img
from .vision import VisionDataset


class Cloud38(VisionDataset):
    """Cloud 38 Dataset.

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

    mirrors = "http://vault.sfu.ca/index.php/s/pymNqYF09JkM8Bp/download"
    resources = "38cloud.zip"

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:

        super(Cloud38, self).__init__(root, transform=transform, target_transform=target_transform)

        self.root = root
        self.data_path = os.path.join(self.root, "38cloud")
        self.base_path = Path(os.path.join(self.data_path, "38-Cloud_training"))

        if not os.path.exists(self.root):
            os.makedirs(self.root)

        if download and self._check_exists():
            print("file already exists.")

        if download and not self._check_exists():
            self.download()
            self.extract_file()

        self.file_validator()
        self.labels = self.get_path()
        print("Done.")

    def file_validator(self):
        if not (self.base_path / "train_rgb").exists():
            (self.base_path / "train_rgb").mkdir()

        if not (self.base_path / "labels").exists():
            (self.base_path / "labels").mkdir()

        for red_patch in (self.base_path / "train_red").iterdir():
            self.create_rgb_pil(red_patch)

        for gt_patch in (self.base_path / "train_gt").iterdir():
            self.convert_tif_png(gt_patch, self.base_path / "labels")

    def get_path(self):
        label = []
        path_label = os.path.join(self.base_path, "labels")
        path_gt = os.path.join(self.base_path, "train_gt")
        label_listing = [os.path.join(path_label, i) for i in os.listdir(path_label)]
        gt_listing = [os.path.join(path_gt, i) for i in os.listdir(path_gt)]
        return pd.DataFrame({"GT": gt_listing, "Label": label_listing})

    def create_rgb_pil(self, red_filename: Path):
        """Combining three bands to RGB format"""
        self.red_filename = str(red_filename)
        green_fn = self.red_filename.replace("red", "green")
        blue_fn = self.red_filename.replace("red", "blue")
        rgb_fn = self.red_filename.replace("red", "rgb").replace(".TIF", ".png")

        array_red = np.array(Image.open(self.red_filename))
        array_green = np.array(Image.open(green_fn))
        array_blue = np.array(Image.open(blue_fn))

        array_rgb = np.stack([array_red, array_green, array_blue], axis=2)
        array_rgb = array_rgb / np.iinfo(array_rgb.dtype).max

        rgb = Image.fromarray((256 * array_rgb).astype(np.uint8), "RGB")
        rgb.save(rgb_fn)
        return rgb

    def convert_tif_png(self, tif_file: Path, out_folder: Path):
        """Converting TIF file to PNG format"""
        self.tif_file = tif_file
        self.out_folder = out_folder
        array_tif = np.array(Image.open(self.tif_file))
        im = Image.fromarray(np.where(array_tif == 255, 1, 0))
        im.save(self.out_folder / self.tif_file.with_suffix(".png").name)
        return im

    def __len__(self) -> int:
        return len(self.img_labels)

    def download(self) -> None:
        """download and extract file."""
        _urlretrieve(self.mirrors, os.path.join(self.root, self.resources))

    def _check_exists(self):
        """Check file has been download or not"""
        folders = [
            "38-Cloud_95-Cloud_Test_Metadata_Files",
            "38-Cloud_test",
            "38-Cloud_training",
            "38-Cloud_Training_Metadata_Files",
        ]

        status = [
            os.path.exists(os.path.join(self.data_path, folder_pth)) for folder_pth in folders
        ]
        return all(status)

    def extract_file(self):
        """Extract file from the compressed"""
        print("Extracting...")
        shutil.unpack_archive(os.path.join(self.root, self.resources), self.root)
        os.remove(os.path.join(self.root, self.resources))
