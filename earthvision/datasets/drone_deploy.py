"""Drone Deploy Dataset - Semantic Segmentation."""
from PIL import Image
import sys
import os
import numpy as np
import random
import cv2

from typing import Any, Callable, Optional, Tuple
from .vision import VisionDataset
from earthvision.constants.DroneDeploy.config import (
    train_ids,
    val_ids,
    test_ids,
    LABELMAP,
    INV_LABELMAP,
)
from earthvision.datasets.utils import _urlretrieve


class DroneDeploy(VisionDataset):
    """Drone Deploy Semantic Dataset.

    Args:
        root (string): Root directory of dataset.
        dataset_type (string, optional): Choose dataset type.
        data_mode (int): 0 for train data, 1 for validation data, and 2 for testing data
        transform (callable, optional): A function/transform that  takes in an PIL image and
            returns a transformed version. E.g, transforms.RandomCrop
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """

    resources = {
        "dataset-sample": "https://dl.dropboxusercontent.com/s/h8a8kev0rktf4kq/dataset-sample.tar.gz?dl=0",
        "dataset-medium": "https://dl.dropboxusercontent.com/s/r0dj9mhyv4bgbme/dataset-medium.tar.gz?dl=0",
    }

    def __init__(
        self,
        root: str,
        dataset_type="dataset-sample",
        data_mode: int = 0,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:

        super(DroneDeploy, self).__init__(
            root, transform=transform, target_transform=target_transform
        )

        self.root = root
        self.dataset_type = dataset_type
        self.filename = f"{dataset_type}.tar.gz"
        self.filepath = os.path.join(self.root, self.filename)
        self.data_mode = data_mode
        self.label_path = f"{dataset_type}/label-chips"
        self.image_path = f"{dataset_type}/image-chips"

        if download and self._check_exists():
            print("file already exists.")

        if download and not self._check_exists():
            self.download()

        self.load_dataset()

    def download(self) -> None:
        """Download a dataset, extract it and create the tiles."""
        print(f'Downloading "{self.dataset_type}"')
        self.root = os.path.expanduser(self.root)
        fpath = os.path.join(self.root, self.filename)
        _urlretrieve(self.resources[self.dataset_type], fpath)

        if not os.path.exists(os.path.join(self.root, self.dataset_type)):
            print(f'Extracting "{self.filepath}"')
            os.system(f"tar -xvf {self.filepath}")
            os.system(f"mv {self.dataset_type} {self.root}")
        else:
            print(f'Folder "{self.dataset_type}" already exists.')

        image_chips = f"{self.dataset_type}/image-chips"
        label_chips = f"{self.dataset_type}/label-chips"

        if not os.path.exists(image_chips):
            os.mkdir(os.path.join(self.root, image_chips))
        if not os.path.exists(label_chips):
            os.mkdir(os.path.join(self.root, label_chips))

        run(os.path.join(self.root, self.dataset_type))

    def _check_exists(self) -> bool:
        if self.dataset_type not in self.resources.keys():
            print(f"Unknown dataset {self.dataset_type}")
            print(f"Available dataset : {self.resources.keys()}")
            sys.exit(0)

        if os.path.exists(self.filepath):
            return True
        else:
            return False

    def load_dataset(self):
        if self.data_mode == 0:
            list_chip = "train.txt"
        elif self.data_mode == 1:
            list_chip = "valid.txt"
        elif self.data_mode == 2:
            list_chip = "test.txt"

        files = [
            f"{os.path.join(self.root, self.dataset_type)}/image-chips/{fname}"
            for fname in load_lines(os.path.join(self.root, self.dataset_type, list_chip))
        ]
        self.image_files = files

    def __getitem__(self, idx) -> Tuple[Any, Any]:
        """
        Args:
            idx (int): Index
        Returns:
            tuple: (img, target) where target is index of the target class.
        """
        image_file = self.image_files[idx]
        label_file = image_file.replace(self.image_path, self.label_path)

        img = np.array(load_img(image_file))
        target = mask_to_classes(load_img(label_file))
        target = np.array(target)

        if self.transform is not None:
            img = Image.fromarray(img)
            img = self.transform(img)

        if self.target_transform is not None:
            target = Image.fromarray(target)
            target = self.target_transform(target)
        return img, target

    def __len__(self) -> int:
        return len(self.image_files)

    def on_epoch_end(self):
        random.shuffle(self.image_files)


def load_lines(fname):
    with open(fname, "r") as f:
        return [line.strip() for line in f.readlines()]


def load_img(fname):
    return np.array(Image.open(fname))


def mask_to_classes(mask):
    return to_categorical(mask[:, :, 0], 6)


def to_categorical(y, num_classes=None, dtype="float32"):
    """Converts a class vector (integers) to binary class matrix.
    E.g. for use with categorical_crossentropy.
    Args:
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes. If `None`, this would be inferred
          as the (largest number in `y`) + 1.
        dtype: The data type expected by the input. Default: `'float32'`.
    Returns:
        A binary matrix representation of the input. The classes axis is placed
        last.
    Raises:
        Value Error: If input contains string value
    """
    y = np.array(y, dtype="int")
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


def get_split(scene):
    if scene in train_ids:
        return "train.txt"
    if scene in val_ids:
        return "valid.txt"
    if scene in test_ids:
        return "test.txt"


def color2class(orthochip, img):
    ret = np.zeros((img.shape[0], img.shape[1]), dtype="uint8")
    ret = np.dstack([ret, ret, ret])
    colors = np.unique(img.reshape(-1, img.shape[2]), axis=0)

    # Skip any chips that would contain magenta (IGNORE) pixels
    seen_colors = set([tuple(color) for color in colors])
    IGNORE_COLOR = LABELMAP[0]
    if IGNORE_COLOR in seen_colors:
        return None, None

    for color in colors:
        locs = np.where(
            (img[:, :, 0] == color[0]) & (img[:, :, 1] == color[1]) & (img[:, :, 2] == color[2])
        )
        ret[locs[0], locs[1], :] = INV_LABELMAP[tuple(color)] - 1

    return orthochip, ret


def image2tile(
    prefix,
    scene,
    dataset,
    orthofile,
    elevafile,
    labelfile,
    windowx,
    windowy,
    stridex,
    stridey,
):

    ortho = cv2.imread(orthofile)
    label = cv2.imread(labelfile)

    assert ortho.shape[0] == label.shape[0]
    assert ortho.shape[1] == label.shape[1]

    shape = ortho.shape
    xsize = shape[1]
    ysize = shape[0]
    print(f"converting {dataset} image {orthofile} {xsize}x{ysize} to chips ...")

    counter = 0
    for xi in range(0, shape[1] - windowx, stridex):
        for yi in range(0, shape[0] - windowy, stridey):
            orthochip = ortho[yi : yi + windowy, xi : xi + windowx, :]
            labelchip = label[yi : yi + windowy, xi : xi + windowx, :]

            orthochip, classchip = color2class(orthochip, labelchip)

            if classchip is None:
                continue

            orthochip_filename = os.path.join(
                prefix, "image-chips", scene + "-" + str(counter).zfill(6) + ".png"
            )
            labelchip_filename = os.path.join(
                prefix, "label-chips", scene + "-" + str(counter).zfill(6) + ".png"
            )

            with open(f"{prefix}/{dataset}", mode="a") as fd:
                fd.write(scene + "-" + str(counter).zfill(6) + ".png\n")

            cv2.imwrite(orthochip_filename, orthochip)
            cv2.imwrite(labelchip_filename, classchip)
            counter += 1


def run(prefix, size=300, stride=300):
    lines = [line for line in open(f"{prefix}/index.csv")]
    print(
        "converting images to chips - this may take a few minutes but only needs to be done once."
    )

    for lineno, line in enumerate(lines):
        line = line.strip().split(" ")
        scene = line[1]
        dataset = get_split(scene)

        orthofile = os.path.join(prefix, "images", scene + "-ortho.tif")
        elevafile = os.path.join(prefix, "elevations", scene + "-elev.tif")
        labelfile = os.path.join(prefix, "labels", scene + "-label.png")

        if os.path.exists(orthofile) and os.path.exists(labelfile):
            image2tile(
                prefix,
                scene,
                dataset,
                orthofile,
                elevafile,
                labelfile,
                windowx=size,
                windowy=size,
                stridex=stride,
                stridey=stride,
            )
