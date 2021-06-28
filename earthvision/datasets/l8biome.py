import os
import shutil
import posixpath
import numpy as np
import pandas as pd
import glob
import torch
from torch.utils.data import Dataset
from PIL import Image
import requests
from bs4 import BeautifulSoup
from .utils import _urlretrieve, _load_img


class L8Biome():
    """L8 Biome Cloud Cover
    Download page https://landsat.usgs.gov/landsat-8-cloud-cover-assessment-validation-data
    """

    mirrors = 'https://landsat.usgs.gov/landsat-8-cloud-cover-assessment-validation-data'

    def __init__(self, root: str):

        self.root = root
        self.download_urls = self.get_download_url()
        self.resources = [url.split('/')[-1] for url in self.download_urls]
        self.data_modes = [filename.split('.tar.gz')[0]
                           for filename in self.resources]

        if not self._check_exists():
            self.download()
            self.extract_file()

        self.img_labels = self.get_path_and_label()

    def get_download_url(self):
        """Get the urls to download the files."""
        page = requests.get(self.mirror)
        soup = BeautifulSoup(page.content, 'html.parser')

        urls = [url.get('href') for url in soup.find_all('a')]

        download_urls = filter(lambda url: url.endswith('.tar.gz'), urls)
        return download_urls

    def download(self):
        """Download file"""
        for resource in self.resources:
            file_url = posixpath.join(self.mirrors, resource)
            _urlretrieve(file_url, os.path.join(self.root, resource))

    def extract_file(self):
        """Extract the .zip file"""
        for resource in self.resources:
            shutil.unpack_archive(os.path.join(self.root, resource), self.root)
            os.remove(os.path.join(self.root, resource))

    def _check_exists(self):
        is_exists = []
        if not os.path.isdir(self.root):
            os.mkdir(self.root)

        for data_mode in self.data_modes:
            data_path = os.path.join(self.root, "BC", data_mode)
            is_exists.append(os.path.exists(data_path))

        return all(is_exists)

    def get_path_and_label(self):
        """Get the path of the images and labels (masks) in a dataframe"""
        image_path = []
        label = []

        for data_mode in self.data_modes:
            for image in glob.glob(os.path.join(self.root, data_mode, 'LC*.TIF')):
                image_path.append(image)

                label.extend(glob.glob(os.path.join(
                    self.root, data_mode, '*mask.hdr')))

        df = pd.DataFrame({'image': image_path, 'label': label})
        return df

    def __getitem__(self, idx):
        """Return a tensor image and its tensor mask"""
        img_path = self.img_labels.iloc[idx, 0]
        mask_path = self.img_labels.iloc[idx, 1]

        image = _load_img(img_path)
        image = np.array(image)
        image = torch.from_numpy(image)

        mask = _load_img(mask_path)
        mask = np.array(mask)
        mask = torch.from_numpy(mask)

        sample = (image, mask)

        return sample

    def __len__(self):
        return len(self.img_labels)

    def __iter__(self):
        for index in range(self.__len__()):
            yield self.__getitem__(index)
