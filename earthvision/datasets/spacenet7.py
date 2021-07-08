"""Class for SpaceNet 7: Multi-Temporal Urban Development Challenge - Instance Segmentation."""
import os
import shutil
import posixpath
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Resize
from PIL import Image
from .utils import downloader


class SpaceNet7(Dataset):
    """SpaceNet7
    SN7: Multi-Temporal Urban Development Challenge
    <https://spacenet.ai/sn7-challenge/>

    Args:
        root (string): Root directory of dataset.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        data_mode (int): 0 for train data, and 1 for test data.
    """

    resources = {
        'train': 'https://storage.googleapis.com/ossjr/sample_SN7_buildings_train.tar.gz',
        'test': 'https://storage.googleapis.com/ossjr/sample_SN7_buildings_test_public.tar.gz'}

    def __init__(self, root: str, download: bool = False, data_mode: str = 'train'):
        self.root = root
        self.data_mode = data_mode
        self.filename = self.resources.get(data_mode, 'NULL').split('/')[-1]
        self.dataset_path = os.path.join(root, self.filename)
        data_mode_folder = {'train': 'train', 'test': 'test_public'}
        self.folder_name = data_mode_folder.get(data_mode, 'NULL')

        if download:
            if self._check_exists(self.dataset_path):
                raise ValueError("Raw data already exists.")
            else:
                self.download()

        if not self._check_exists(self.folder_name):
            self.extract_file()
        else:
            print("Data already extracted.")

    def _check_exists(self, obj) -> bool:
        if os.path.exists(obj):
            return True
        else:
            return False

    def download(self):
        """Download dataset and extract it"""
        if self.data_mode not in self.resources.keys():
            raise ValueError("Unrecognized data_mode")

        downloader(self.resources[self.data_mode], self.root)

    def extract_file(self):
        shutil.unpack_archive(self.dataset_path, self.root)
