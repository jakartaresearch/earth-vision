"""Class for SpaceNet 7: Multi-Temporal Urban Development Challenge - Instance Segmentation."""
from logging import root
import os
import sys
import shutil
import posixpath
import numpy as np
import pandas as pd
import torch
import multiprocessing
import skimage

from torch.utils.data import Dataset
from torchvision.transforms import Resize
from PIL import Image
from .utils import downloader, _load_img
from .spacenet7_utils import map_wrapper, make_geojsons_and_masks


class SpaceNet7(Dataset):
    """SpaceNet7
    SN7: Multi-Temporal Urban Development Challenge
    <https://spacenet.ai/sn7-challenge/>

    Args:
        root (string): Root directory of dataset.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        data_mode (string): 'train' for train data and 'test' for test data.
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

        if not self._check_exists(os.path.join(self.root, self.folder_name)):
            self.extract_file()
        else:
            print("Data already extracted.")

        if self.data_mode == 'train':
            # TODO: Check 'masks' folder di train data, if masks not available do generate_mask()
            print('Generate label image mask from train data...')
            self.generate_mask()

        self.img_labels = self.get_path_and_label()

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

    def generate_mask(self):
        """
        Create Training Masks
        Multi-thread to increase speed
        We'll only make a 1-channel mask for now, but Solaris supports a multi-channel mask as well, see
            https://github.com/CosmiQ/solaris/blob/master/docs/tutorials/notebooks/api_masks_tutorial.ipynb
        """
        aois = sorted([f for f in os.listdir(os.path.join(self.root, 'train'))
                       if os.path.isdir(os.path.join(self.root, 'train', f))])
        params = []
        make_fbc = False

        input_args = []
        for i, aoi in enumerate(aois):
            print(i, "aoi:", aoi)
            im_dir = os.path.join(self.root, 'train', aoi, 'images_masked/')
            json_dir = os.path.join(self.root, 'train', aoi, 'labels_match/')
            out_dir_mask = os.path.join(self.root, 'train', aoi, 'masks/')
            out_dir_mask_fbc = os.path.join(
                self.root, 'train', aoi, 'masks_fbc/')
            os.makedirs(out_dir_mask, exist_ok=True)
            if make_fbc:
                os.makedirs(out_dir_mask_fbc, exist_ok=True)

            json_files = sorted([f
                                 for f in os.listdir(os.path.join(json_dir))
                                 if f.endswith('Buildings.geojson') and os.path.exists(os.path.join(json_dir, f))])
            for j, f in enumerate(json_files):
                # print(i, j, f)
                name_root = f.split('.')[0]
                json_path = os.path.join(json_dir, f)
                image_path = os.path.join(
                    im_dir, name_root + '.tif').replace('labels', 'images').replace('_Buildings', '')
                output_path_mask = os.path.join(
                    out_dir_mask, name_root + '.tif')
                if make_fbc:
                    output_path_mask_fbc = os.path.join(
                        out_dir_mask_fbc, name_root + '.tif')
                else:
                    output_path_mask_fbc = None

                if (os.path.exists(output_path_mask)):
                    continue
                else:
                    input_args.append([make_geojsons_and_masks,
                                       name_root, image_path, json_path,
                                       output_path_mask, output_path_mask_fbc])

        p = multiprocessing.Pool(multiprocessing.cpu_count() - 1)
        out = p.map(map_wrapper, input_args)
        p.close()
        p.join()

    def get_path_and_label(self):
        """Return dataframe type consist of image path and corresponding label (for train data),
        or image path only (for test data)."""
        pops = ['train', 'test_public']

        for pop in pops:
            d = os.path.join(self.root, pop)
            im_list, mask_list = [], []
            subdirs = sorted([f for f in os.listdir(
                d) if os.path.isdir(os.path.join(d, f))])
            for subdir in subdirs:

                if pop == 'train':
                    im_files = [os.path.join(d, subdir, 'images_masked', f)
                                for f in sorted(os.listdir(os.path.join(d, subdir, 'images_masked')))
                                if f.endswith('.tif') and os.path.exists(os.path.join(d, subdir, 'masks', f.split('.')[0] + '_Buildings.tif'))]
                    mask_files = [os.path.join(d, subdir, 'masks', f.split('.')[0] + '_Buildings.tif')
                                  for f in sorted(os.listdir(os.path.join(d, subdir, 'images_masked')))
                                  if f.endswith('.tif') and os.path.exists(os.path.join(d, subdir, 'masks', f.split('.')[0] + '_Buildings.tif'))]
                    im_list.extend(im_files)
                    mask_list.extend(mask_files)

                elif pop == 'test_public':
                    im_files = [os.path.join(d, subdir, 'images_masked', f)
                                for f in sorted(os.listdir(os.path.join(d, subdir, 'images_masked')))
                                if f.endswith('.tif')]
                    im_list.extend(im_files)

            if self.data_mode == 'train':
                df = pd.DataFrame({'image': im_list, 'label': mask_list})
            elif self.data_mode == 'test':
                df = pd.DataFrame({'image': im_list})

            return df

    def __getitem__(self, idx):
        """Return a tensor image and its tensor mask"""

        img_path = self.img_labels.iloc[idx, 0]
        image = _load_img(img_path)
        image = np.array(image)
        image = torch.from_numpy(image)

        if self.data_mode == 'train':
            mask_path = self.img_labels.iloc[idx, 1]
            mask = _load_img(mask_path)
            mask = np.array(mask)
            mask = torch.from_numpy(mask)
            sample = (image, mask)
        elif self.data_mode == 'test':
            sample = (image)

        return sample

    def __len__(self):
        return len(self.img_labels)
