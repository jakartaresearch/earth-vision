"""Class for Aerial Cactus - Image Classification."""
import sys
import os
import numpy as np
import random
import torch
from .utils import _urlretrieve

from PIL import Image
from torch.utils.data import Dataset


class AerialCactus():
    """Aerial Cactus Image Classification. `Download Aerial Cactus <https://www.kaggle.com/irvingvasquez/cactus-aerial-photos>`_

    Args:
        root (string): Root directory of dataset.
        data_mode (int): 0 for train data, 1 for validation data
    """

    def __init__(self, root: str, data_mode: int = 0):
        self.root = root
        self.training_path = f'{root}/training_set/training_set'
        self.validation_path = f'{root}/validation_set/validation_set'
        self.data_mode = data_mode

        assert self._check_exists(), "Ensure you have the right dataset."

        self.load_dataset()

    def _check_exists(self) -> bool:
        if os.path.exists(self.training_path) \
                and os.path.exists(self.validation_path) \
                and os.path.exists(os.path.join(self.training_path, 'cactus')) \
                and os.path.exists(os.path.join(self.training_path, 'no_cactus')) \
                and os.path.exists(os.path.join(self.validation_path, 'cactus')) \
                and os.path.exists(os.path.join(self.validation_path, 'no_cactus')):
            return True
        else:
            return False

    def load_dataset(self):
        if self.data_mode == 0:
            files = [
                fname for fname in os.listdir(self.training_path)]
        elif self.data_mode == 1:
            files = [
                fname for fname in os.listdir(self.validation_path)]
        self.image_files = files

    # def __len__(self):
    #     return len(self.image_files)

    # def __getitem__(self, idx):
    #     image_file = self.image_files[idx]
    #     label_file = image_file.replace(self.image_path, self.label_path)

    #     image = load_img(image_file)
    #     label = mask_to_classes(load_img(label_file))

    #     tensor_image = torch.from_numpy(np.array(image))
    #     tensor_label = torch.from_numpy(np.array(label))
    #     return tensor_image, tensor_label

    # def on_epoch_end(self):
    #     random.shuffle(self.image_files)


def load_img(fname):
    return np.array(Image.open(fname))


def mask_to_classes(mask):
    return to_categorical(mask[:, :, 0], 6)
