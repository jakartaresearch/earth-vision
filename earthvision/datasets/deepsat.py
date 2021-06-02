"""Class for Deepsat - Scene Classification."""
import os
import scipy.io as sio
import torch

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

    resources = {'SAT-4_and_SAT-6_datasets': ''}

    def __init__(self, root: str, dataset_type='SAT-4', download: bool = False, data_mode: int = 0):
        self.root = root
        self.dataset_type = dataset_type
        self.data_mode = data_mode

        dataset = self.load_dataset()
        self.choose_data_mode(dataset)

    def load_dataset(self):
        folder_pth = list(self.resources.keys())[0]
        filename = {'SAT-4': 'sat-4-full.mat', 'SAT-6': 'sat-6-full.mat'}
        dataset = sio.loadmat(os.path.join(
            folder_pth, filename[self.dataset_type]))
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
