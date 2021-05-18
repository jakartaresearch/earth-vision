import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import Resize
import torch

class AerialCactus():
    """Aerial Cactus Dataset.

    <https://www.kaggle.com/c/aerial-cactus-identification>
    """

    def __init__(self, root, data_mode='training_set', transform=Resize((32, 32)), target_transform=None):
        self.root = root
        self.data_mode = data_mode
        self.img_labels = self.get_path_and_label()
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.img_labels.iloc[idx, 0]
        label = self.img_labels.iloc[idx, 1]
        image = load_img(img_path)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        image = np.array(image)
        image = torch.from_numpy(image)
        sample = {"image": image, "label": label}
        return sample
    
    def get_path_and_label(self):
        classes = {'cactus':1, 'no_cactus':0}
        image_path = []
        label = []
        for cat, enc in classes.items():
            cat_path = os.path.join(self.root, 'archive', self.data_mode, self.data_mode, cat)
            cat_image = [os.path.join(cat_path, path) for path in os.listdir(cat_path)]
            cat_label = [enc] * len(cat_image)
            image_path += cat_image
            label += cat_label
        df = pd.DataFrame({'image':image_path, 'label':label})
        return df

    def _check_exists(self):
        self.train_path = os.path.join(
            self.root, "cactus-aerial-photos", "training_set", "training_set")
        self.test_path = os.path.join(
            self.root, "cactus-aerial-photos", "validation_set", "validation_set")

        return os.path.exists(os.path.join(self.train_path, "cactus")) and \
            os.path.exists(os.path.join(self.train_path, "no_cactus")) and \
            os.path.exists(os.path.join(self.test_path, "cactus")) and \
            os.path.exists(os.path.join(self.test_path, "no_cactus"))

    def download(self):
        """Download and extract file."""
        file_url = posixpath.join(self.mirrors, self.resources)
        _urlretrieve(file_url, self.resources)

    def extract_file(self):
        """Extract file from compressed."""
        path_destination = os.path.join(
            self.root, self.resources.replace(".zip", ""))
        os.makedirs(path_destination, exist_ok=True)
        shutil.unpack_archive(self.resources, f"{path_destination}")
        os.remove(self.resources)
