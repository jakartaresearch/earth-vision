"""Cars Overhead with Context."""
import os
import shutil
import posixpath
import tarfile
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from earthvision.datasets.utils import _urlretrieve, _load_img
from earthvision.constants.COWC.config import file_mapping_counting, \
    file_mapping_detection


class COWC():
    """ Cars Overhead with Context.
    https://gdo152.llnl.gov/cowc/
    """

    mirrors = "https://gdo152.llnl.gov/cowc/download"
    resources = "cowc-everything.txz"

    def __init__(self,
                 root: str,
                 data_mode: str = 'train',
                 task_mode: str = 'counting',
                 transform=None,
                 target_transform=None):
        self.root = root
        self.data_mode = data_mode
        self.task_mode = task_mode
        self.transform = transform
        self.target_transform = target_transform

        if not self._check_exists():
            self.download()
            self.extract_file()

        if self.task_mode == 'counting':
            self.task_path = os.path.join(
                self.root, 'cowc/datasets/patch_sets/counting'
            )
            self.file_mapping = file_mapping_counting
        elif self.task_mode == 'detection':
            self.task_path = os.path.join(
                self.root, 'cowc/datasets/patch_sets/detection'
            )
            self.file_mapping = file_mapping_detection
        else:
            raise ValueError('task_mode not recognized.')

        for filename, compressed in self.file_mapping.items():
            if not self._check_exists_subfile(filename):
                self.extract_subfile(filename, compressed)

        self.img_labels = self.get_path_and_label()

    def __getitem__(self, idx):
        img_path = self.img_labels.iloc[idx, 0]
        label = self.img_labels.iloc[idx, 1]
        folder = img_path.split('/', 1)[0]
        img_path = os.path.join(self.task_path, folder, img_path)
        image = _load_img(img_path)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        image = np.array(image)
        image = torch.from_numpy(image)
        sample = (image, label)

        return sample

    def __len__(self):
        return len(self.img_labels)

    def get_path_and_label(self):
        """Return dataframe type consist of image path
        and corresponding label."""

        if self.task_mode == 'counting':
            if self.data_mode == 'train':
                label_name = 'COWC_train_list_64_class.txt.bz2'
            elif self.data_mode == 'test':
                label_name = 'COWC_test_list_64_class.txt.bz2'
            else:
                raise ValueError('data_mode not recognized.')
        elif self.task_mode == 'detection':
            if self.data_mode == 'train':
                label_name = 'COWC_train_list_detection.txt.bz2'
            elif self.data_mode == 'test':
                label_name = 'COWC_test_list_detection.txt.bz2'
            else:
                raise ValueError('data_mode not recognized.')
        else:
            raise ValueError('task_mode not recognized.')

        label_path = os.path.join(self.task_path, label_name)
        df = pd.read_csv(label_path, sep=' ', header=None)

        return df

    def _check_exists_subfile(self, filename):
        path_to_check = os.path.join(self.task_path, filename)
        return os.path.exists(path_to_check)

    def extract_subfile(self, filename, compressed):
        comp_path = os.path.join(self.task_path, compressed)
        file_path = os.path.join(self.task_path, filename)
        tar = tarfile.open(comp_path)
        tar.extractall(file_path)
        tar.close()

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, "cowc"))

    def download(self):
        """download file."""
        file_url = posixpath.join(self.mirrors, self.resources)
        _urlretrieve(file_url, os.path.join(self.root, self.resources))

    def extract_file(self):
        """Extract file from compressed."""
        shutil.unpack_archive(os.path.join(
            self.root, self.resources), self.root)
        os.remove(os.path.join(self.root, self.resources))
