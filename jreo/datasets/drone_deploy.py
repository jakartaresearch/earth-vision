"""Class for Drone Deploy - Semantic Segmentation."""
import sys
import os
import numpy as np
import random
import torch
from .images2chips import run
from .utils import _urlretrieve, to_categorical

from PIL import Image
from torch.utils.data import Dataset


class DroneDeploy():
    """Drone Deploy Semantic Dataset.

    Args:
        root (string): Root directory of dataset.
        dataset_type (string, optional): Choose dataset type.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        data_mode (int): 0 for train data, 1 for validation data, and 2 for testing data
    """

    resources = {
        'dataset-sample': 'https://dl.dropboxusercontent.com/s/h8a8kev0rktf4kq/dataset-sample.tar.gz?dl=0',
        'dataset-medium': 'https://dl.dropboxusercontent.com/s/r0dj9mhyv4bgbme/dataset-medium.tar.gz?dl=0'
    }

    def __init__(self, root: str, dataset_type='dataset-sample', download: bool = False, data_mode: int = 0):
        self.root = root
        self.dataset_type = dataset_type
        self.filename = f'{dataset_type}.tar.gz'
        self.data_mode = data_mode
        self.label_path = f'{dataset_type}/label-chips'
        self.image_path = f'{dataset_type}/image-chips'

        if download and self._check_exists():
            print(f'zipfile "{self.filename}" already exists.')

        if download and not self._check_exists():
            self.download()

        self.load_dataset()

    def download(self):
        """Download a dataset, extract it and create the tiles."""
        print(f'Downloading "{self.dataset_type}"')
        self.root = os.path.expanduser(self.root)
        fpath = os.path.join(self.root, self.filename)
        _urlretrieve(self.resources[self.dataset_type], fpath)

        if not os.path.exists(self.dataset_type):
            print(f'Extracting "{self.filename}"')
            os.system(f'tar -xvf {self.filename}')
        else:
            print(f'Folder "{self.dataset_type}" already exists.')

        image_chips = f'{self.dataset_type}/image-chips'
        label_chips = f'{self.dataset_type}/label-chips'

        if not os.path.exists(image_chips):
            os.mkdir(image_chips)
        if not os.path.exists(label_chips):
            os.mkdir(label_chips)

        run(self.dataset_type)

    def _check_exists(self) -> bool:
        if self.dataset_type not in self.resources.keys():
            print(f"Unknown dataset {self.dataset_type}")
            print(f"Available dataset : {self.resources.keys()}")
            sys.exit(0)

        if os.path.exists(self.filename):
            return True
        else:
            return False

    def load_dataset(self):
        if self.data_mode == 0:
            list_chip = 'train.txt'
        elif self.data_mode == 1:
            list_chip = 'valid.txt'
        elif self.data_mode == 2:
            list_chip = 'test.txt'

        files = [f'{self.dataset_type}/image-chips/{fname}'
                 for fname in load_lines(os.path.join(self.dataset_type, list_chip))]
        self.image_files = files

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        label_file = image_file.replace(self.image_path, self.label_path)

        image = load_img(image_file)
        label = mask_to_classes(load_img(label_file))

        tensor_image = torch.from_numpy(np.array(image))
        tensor_label = torch.from_numpy(np.array(label))
        return tensor_image, tensor_label

    def on_epoch_end(self):
        random.shuffle(self.image_files)


def load_lines(fname):
    with open(fname, 'r') as f:
        return [line.strip() for line in f.readlines()]


def load_img(fname):
    return np.array(Image.open(fname))


def mask_to_classes(mask):
    return to_categorical(mask[:, :, 0], 6)
