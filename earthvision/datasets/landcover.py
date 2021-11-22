"""The LandCover.ai (Land Cover from Aerial Imagery) Dataset."""
from PIL import Image
import os
import shutil
import posixpath
import numpy as np
import pandas as pd
import glob
import cv2

from typing import Any, Callable, Optional, Tuple
from .utils import _urlretrieve, _load_img
from .vision import VisionDataset
from torchvision.transforms import Resize, ToTensor, Compose


class LandCover(VisionDataset):
    """The LandCover.ai (Land Cover from Aerial Imagery) dataset.

    <https://landcover.ai/download/landcover.ai.v1.zip>

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

    mirrors = "https://landcover.ai/download"
    resources = "landcover.ai.v1.zip"

    def __init__(
        self,
        root: str,
        transform=Compose([Resize((256, 256)), ToTensor()]),
        target_transform=Compose([Resize((256, 256)), ToTensor()]),
        download: bool = False,
    ) -> None:

        super(LandCover, self).__init__(
            root, transform=transform, target_transform=target_transform
        )

        self.root = root

        if download and self._check_exists():
            print("file already exists.")

        if download and not self._check_exists():
            self.download()
            self.extract_file()
            self.to_chip_img_mask("landcover")

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
        img = np.array(_load_img(img_path))
        mask = np.array(_load_img(mask_path))

        if self.transform is not None:
            img = Image.fromarray(img)
            img = self.transform(img)

        if self.target_transform is not None:
            mask = Image.fromarray(mask)
            mask = self.target_transform(mask)
        return img, mask

    def __len__(self) -> int:
        return len(self.img_labels)

    def get_image_path_and_mask_path(self):
        """Return dataframe type consist of image path and mask path."""

        img_path = os.path.join(self.root, "landcover", "images")
        msk_path = os.path.join(self.root, "landcover", "masks")

        images_path = [os.path.join(img_path, path) for path in os.listdir(img_path)]
        images_path.sort()
        masks_path = [os.path.join(img_path, path) for path in os.listdir(msk_path)]
        masks_path.sort()

        df = pd.DataFrame({"image": images_path, "mask": masks_path})
        return df

    def to_chip_img_mask(self, base):
        IMGS_DIR = "./{}/images".format(base)
        MASKS_DIR = "./{}/masks".format(base)
        OUTPUT_DIR = "./{}/output".format(base)
        OUTPUT_IMGS_DIR = "./{}/output/images".format(base)
        OUTPUT_MASKS_DIR = "./{}/output/masks".format(base)

        TARGET_SIZE = 512

        img_paths = glob.glob(os.path.join(IMGS_DIR, "*.tif"))
        mask_paths = glob.glob(os.path.join(MASKS_DIR, "*.tif"))

        img_paths.sort()
        mask_paths.sort()

        # os.makedirs(OUTPUT_DIR)
        os.makedirs(OUTPUT_IMGS_DIR)
        os.makedirs(OUTPUT_MASKS_DIR)
        for i, (img_path, mask_path) in enumerate(zip(img_paths, mask_paths)):
            img_filename = os.path.splitext(os.path.basename(img_path))[0]
            mask_filename = os.path.splitext(os.path.basename(mask_path))[0]
            img = cv2.imread(img_path)
            mask = cv2.imread(mask_path)

            assert img_filename == mask_filename and img.shape[:2] == mask.shape[:2]

            k = 0
            for y in range(0, img.shape[0], TARGET_SIZE):
                for x in range(0, img.shape[1], TARGET_SIZE):
                    img_tile = img[y : y + TARGET_SIZE, x : x + TARGET_SIZE]
                    mask_tile = mask[y : y + TARGET_SIZE, x : x + TARGET_SIZE]

                    if img_tile.shape[0] == TARGET_SIZE and img_tile.shape[1] == TARGET_SIZE:
                        out_img_path = os.path.join(
                            OUTPUT_DIR, "images", "{}_{}.jpg".format(img_filename, k)
                        )
                        cv2.imwrite(out_img_path, img_tile)

                        out_mask_path = os.path.join(
                            OUTPUT_DIR, "masks", "{}_{}.png".format(mask_filename, k)
                        )
                        cv2.imwrite(out_mask_path, mask_tile)

                    k += 1

            print("Processed {} {}/{}".format(img_filename, i + 1, len(img_paths)))

    def download(self) -> None:
        """download and extract file."""
        file_url = posixpath.join(self.mirrors, self.resources)
        _urlretrieve(file_url, os.path.join(self.root, self.resources))

    def _check_exists(self):
        """Check file has been download or not"""
        self.data_path = os.path.join(
            self.root,
            "landcover",
        )

        return os.path.exists(os.path.join(self.data_path, "images")) and os.path.exists(
            os.path.join(self.data_path, "masks")
        )

    def extract_file(self):
        """Extract file from compressed."""
        os.makedirs(os.path.join(self.root, "landcover"))
        shutil.unpack_archive(
            os.path.join(self.root, self.resources), os.path.join(self.root, "landcover")
        )
        os.remove(os.path.join(self.root, self.resources))
