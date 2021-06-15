"""Class for Deepsat - Scene Classification."""
import os
import scipy.io as sio
import torch
import requests
import gdown
import sys

from torch.utils.data import Dataset


class DeepSat(Dataset):
    """DeepSat Dataset.

    Args:
        root (string): Root directory of dataset.
        dataset_type (string, optional): Choose dataset type ['SAT-4', 'SAT-6'].
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        data_mode (int): 0 for train data, and 1 for test data.
    """

    resources = {
        'SAT-4_and_SAT-6_datasets': 'https://drive.google.com/uc?id=0B0Fef71_vt3PUkZ4YVZ5WWNvZWs&export=download'}
    dataset_types = ['SAT-4', 'SAT-6']

    def __init__(self, root: str, dataset_type='SAT-4', download: bool = False, data_mode: int = 0):
        self.root = root
        self.dataset_type = dataset_type
        self.data_mode = data_mode
        self.folder_pth = os.path.join(
            self.root, list(self.resources.keys())[0])
        self.filename = list(self.resources.keys())[0] + '.tar.gz'

        if download and self._check_exists():
            print(f'zipfile "{self.filename}" already exists.')

        if download and not self._check_exists():
            self.download()

        dataset = self.load_dataset()
        self.choose_data_mode(dataset)

    def download(self):
        """Download dataset and extract it"""

        self.root = os.path.expanduser(self.root)
        print("Download dataset...")

        gdown.download(self.resources['SAT-4_and_SAT-6_datasets'],
                       os.path.join(self.root, self.filename), quiet=False)

        if os.path.exists(self.folder_pth):
            print(f'file {self.folder_pth} already exists')
        else:
            os.mkdir(self.folder_pth)
            print(f'Extracting file {self.filename}')
            os.system(
                f'tar -xvf {os.path.join(self.root, self.filename)} -C {self.folder_pth}')
            os.system(f'mv {self.folder_pth} {self.root}')
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
        filename = {'SAT-4': 'sat-4-full.mat', 'SAT-6': 'sat-6-full.mat'}
        dataset = sio.loadmat(os.path.join(
            self.folder_pth, filename[self.dataset_type]))
        return dataset

    def choose_data_mode(self, dataset):
        if self.data_mode == 0:
            x_type, y_type = 'train_x', 'train_y'
        elif self.data_mode == 1:
            x_type, y_type = 'test_x', 'test_y'

        self.x, self.y = dataset[x_type], dataset[y_type]
        self.annot = dataset['annotations']

    def __len__(self):
        return self.x.shape[3]

    def __getitem__(self, idx):
        img = self.x[:, :, :, idx]
        label = self.y[:, idx]
        tensor_image = torch.from_numpy(img)
        tensor_label = torch.from_numpy(label)
        return tensor_image, tensor_label
