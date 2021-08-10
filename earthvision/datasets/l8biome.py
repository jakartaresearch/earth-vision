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
from .utils import _urlretrieve, _load_img,_load_img_hdr,_load_stack_img


class L8Biome():
    """L8 Biome Cloud Cover
    Download page https://landsat.usgs.gov/landsat-8-cloud-cover-assessment-validation-data
    """

    mirrors = 'https://landsat.usgs.gov/landsat-8-cloud-cover-assessment-validation-data'

    def __init__(self, root: str):

        self.root = root
        self.download_urls = self.get_download_url()
        self.data_modes = [url.split("/")[-1] for url in self.download_urls]
        
        if not self._check_exists():
            self.download()
            self.extract_file()

#         self.img_labels = self.get_path_and_label()

    def get_download_url(self):
        """Get the urls to download the files."""
        page = requests.get(self.mirrors)
        soup = BeautifulSoup(page.content, 'html.parser')

        urls = [url.get('href') for url in soup.find_all('a')]

        download_urls = list(filter(lambda url: url.endswith('.tar.gz') if url else None, urls))
        return download_urls

    def download(self):
        """Download file"""
        for resource in self.download_urls:
            filename = resource.split("/")[-1]
            _urlretrieve(resource, os.path.join(self.root, filename))
            break

    def extract_file(self):
        """Extract the .zip file"""
        for resource in self.data_modes:
            shutil.unpack_archive(os.path.join(self.root, resource), self.root)
            os.remove(os.path.join(self.root, resource))
            break
    
    # Yang belum
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
        image_directory = []
        label = []
        for 

        for data_mode in self.data_modes:
            for image_dir in glob.glob(os.path.join(self.root, data_mode)):
                image_directory.append(image_dir)

                label.extend(glob.glob(os.path.join(
                    self.root, data_mode, '*mask.hdr')))

        df = pd.DataFrame({'image': image_directory, 'label': label})
        return df

    def __getitem__(self, idx):
        """Return a tensor image and its tensor mask"""
        img_directory = self.img_labels.iloc[idx, 0]
        mask_path = self.img_labels.iloc[idx, 1]

        ls_stack_path = []
        for idx in range(1,12):
            name_file = f"{img_directory}/{img_directory}_B{idx}.TIF"
            ls_stack_path.append(name_file)

        image = _load_stack_img(ls_stack_path)
        image = torch.from_numpy(image)

        mask = _load_img_hdr(mask_path)
        mask = torch.from_numpy(mask)

        sample = (image, mask)

        return sample

    def __len__(self):
        return len(self.img_labels)

    def __iter__(self):
        for index in range(self.__len__()):
            yield self.__getitem__(index)
