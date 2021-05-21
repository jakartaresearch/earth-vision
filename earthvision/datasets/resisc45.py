"""RESISC45 Dataset."""
import os
import posixpath
import shutil
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from ..constants.RESISC45.config import CLASS_ENC
from .utils import _urlretrieve, _to_categorical, _load_img


class RESISC45(Dataset):

    mirrors = "https://storage.googleapis.com/ossjr"
    resources = "NWPU-RESISC45.zip"

    def __init__(self, root, transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.class_enc = CLASS_ENC

        if not self._check_exists():
            self.download()
            self.extract_file()

        self.img_labels = self.get_path_and_label()

    def __getitem__(self, idx):
        img_path = self.img_labels.iloc[idx, 0]
        label = self.img_labels.iloc[idx, 1]
        image = _load_img(img_path)
        label = _to_categorical(label, len(self.class_enc))
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

    def __iter__(self):
        for index in range(self.__len__()):
            yield self.__getitem__(index)

    def get_path_and_label(self):
        """Return dataframe type consist of image path and corresponding label."""
        DATA_SIZE = 700
        category = os.listdir(os.path.join(self.root, 'NWPU-RESISC45'))
        image_path = []
        label = []
        for cat in category:
            cat_enc = self.class_enc[cat]
            label += [cat_enc] * DATA_SIZE
            for num in range(1, DATA_SIZE+1):
                filename = cat + '_' + str(num).zfill(3) + '.jpg'
                image_path += [os.path.join(self.root,
                                            'NWPU-RESISC45', cat, filename)]
        df = pd.DataFrame({'image': image_path, 'label': label})

        return df

    def _check_exists(self):
        is_exists = os.path.exists(os.path.join(self.root, "NWPU-RESISC45"))

        return is_exists

    def download(self):
        """Download and extract file."""
        file_url = posixpath.join(self.mirrors, self.resources)
        _urlretrieve(file_url, os.path.join(self.root, self.resources))

    def extract_file(self):
        """Extract file from compressed."""
        shutil.unpack_archive(os.path.join(
            self.root, self.resources), f"{self.root}")
        os.remove(os.path.join(self.root, self.resources))
