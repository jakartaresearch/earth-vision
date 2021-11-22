"""Dataset from DIUx xView 2018 Detection Challenge."""
from PIL import Image
import os
import shutil
import numpy as np
import json

from typing import Any, Callable, Optional, Tuple
from .utils import _urlretrieve, _load_img
from .vision import VisionDataset
from ..constants.XView.config import index_mapping, CLASS_ENC, CLASS_DEC


class XView(VisionDataset):
    """Dataset from DIUx xView 2018 Detection Challenge.

    Source: https://challenge.xviewdataset.org/data-download (must login)

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

    Samples at:
    - https://storage.googleapis.com/ossjr/xview/train_images.tgz
    - https://storage.googleapis.com/ossjr/xview/train_labels.tgz
    - https://storage.googleapis.com/ossjr/xview/validation_images.tgz
    """

    urls = []
    resources = ["train_images.tgz", "train_labels.tgz", "validation_images.tgz"]

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:

        super(XView, self).__init__(root, transform=transform, target_transform=target_transform)

        self.root = root
        self.data_mode = "train" if train else "validation"
        self.class_enc = CLASS_ENC
        self.class_dec = CLASS_DEC
        self.coords, self.chips, self.classes = None, None, None

        if download and self._check_exists():
            print("file already exists.")

        if download and not self._check_exists():
            self.download()
            self.extract_file()

        if self.data_mode == "train":
            self.coords, self.chips, self.classes = self.get_path_and_label()
            self.imgs = list(os.listdir(os.path.join(self.root, "train_images")))

        elif self.data_mode == "validation":
            self.imgs = list(os.listdir(os.path.join(self.root, "val_images")))

    def _check_exists(self) -> bool:

        if not os.path.isdir(self.root):
            os.mkdir(self.root)

        return (
            os.path.exists(os.path.join(self.root, self.resources[0].split(".")[0]))
            and os.path.exists(os.path.join(self.root, "xView_train.geojson"))
            if self.data_mode == "train"
            else os.path.exists(os.path.join(self.root, "val_images"))
        )

    def download(self):
        """Download file by asking users to input the link"""
        train_images = input(
            "Please follow the following steps to download the required dataset\n"
            + "1. Visit https://challenge.xviewdataset.org/login\n"
            + "2. Sign up for an account\n"
            + "3. Verify your account\n"
            "4. Follow this link: https://challenge.xviewdataset.org/download-links\n"
            "5. Copy the link for 'Download Training Images (tgz)' and paste it: "
        )

        train_labels = input("\n6. Copy and paste the link for 'Download Training Labels (tgz)': ")

        val_images = input("\n7. Copy and paste the link for 'Download Validation Images (tgz)': ")

        self.urls = [train_images, train_labels, val_images]

        for idx, url in enumerate(self.urls):
            _urlretrieve(url, os.path.join(self.root, self.resources[idx]))

    def extract_file(self):
        """Extract the .tgz file"""
        for resource in self.resources:
            shutil.unpack_archive(os.path.join(self.root, resource), self.root)
            os.remove(os.path.join(self.root, resource))

    def _check_exists_label(self, filename):
        """Check whether bounding boxes, image filenames, and labels
        are already extracted from xView_train.geojson
        """
        path_to_check = os.path.join(self.root, filename)
        return path_to_check, os.path.exists(path_to_check)

    def get_path_and_label(self):
        """Gets bounding boxes, image filenames, and labels
        from xView_train.geojson

        Returns:
            coords: coordinates of the bounding boxes
            chips: image file names
            classes: classes for each ground truth
        """
        # check existnce
        coords_path, coords_exists = self._check_exists_label("coords.npy")
        chips_path, chips_exists = self._check_exists_label("chips.npy")
        classes_path, classes_exists = self._check_exists_label("classes.npy")

        # if exist, load and return
        if coords_exists and chips_exists and classes_exists:
            coords = np.load(coords_path)
            chips = np.load(chips_path)
            classes = np.load(classes_path)
            return coords, chips, classes

        # read xView_train.geojson
        fname = os.path.join(self.root, "xView_train.geojson")
        with open(fname) as f:
            data = json.load(f)

        # initialize
        coords, chips, classes = [], [], []

        # extract
        feat_len = len(data["features"])
        img_files = os.listdir(os.path.join(self.root, self.resources[0].split(".")[0]))

        for i in range(feat_len):
            properties = data["features"][i]["properties"]
            b_id = properties["image_id"]
            val = [int(num) for num in properties["bounds_imcoords"].split(",")]

            # type_id 75 and 82 don't belong to any class
            # https://github.com/DIUx-xView/xView1_baseline/issues/3
            if properties["type_id"] not in [75, 82] and b_id in img_files:
                chips.append(b_id)
                classes.append(properties["type_id"])
                coords.append(val)

        # convert to numpy arrays and save
        coords = np.array(coords)
        chips = np.array(chips)
        classes = np.array(classes)
        np.save(coords_path, coords)
        np.save(chips_path, chips)
        np.save(classes_path, classes)
        return coords, chips, classes

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """
        Args:
            idx (int): Index
        Returns:
            tuple: (img, target) where target is a dictionary of target
        consists of bounding boxes and labels.
        """
        if self.data_mode == "train":
            # image
            img_path = os.path.join(self.root, "train_images", self.chips[idx])
            img = np.array(_load_img(img_path))

            if self.transform is not None:
                img = Image.fromarray(img)
                img = self.transform(img)

            # bounding box
            bbox = self.coords[self.chips == self.chips[idx]]
            # label
            label = self.classes[self.chips == self.chips[idx]]
            label = np.vectorize(index_mapping.get)(label)
            # combine bounding box and label
            target = {}
            target["boxes"] = bbox
            target["labels"] = label
            sample = (img, target)

        elif self.data_mode == "validation":
            # image
            img_path = os.path.join(self.root, "val_images", self.imgs[idx])
            img = np.array(_load_img(img_path))

            if self.transform is not None:
                img = Image.fromarray(img)
                img = self.transform(img)

            sample = img
        return sample

    def __len__(self) -> int:
        return len(self.imgs)
