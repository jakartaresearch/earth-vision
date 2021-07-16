import os
import shutil
import posixpath
import numpy as np
import glob
import json
import torch
from .utils import _urlretrieve, _load_img

class XView():
    """Dataset from DIUx xView 2018 Detection Challenge.
    
    Source:
    https://challenge.xviewdataset.org/data-download (must login)

    Args:
        root (string): Root directory of dataset.
        data_mode (string, optional): 'train' for train data, 
            'validation' for validation data, defaults to 'train'

    Samples at:
    https://storage.googleapis.com/ossjr/xview/train_images.tgz
    https://storage.googleapis.com/ossjr/xview/train_labels.tgz
    https://storage.googleapis.com/ossjr/xview/validation_images.tgz
    """

    # TO BE CHANGED
    mirrors = "https://landsat.usgs.gov/cloud-validation/sparcs/"
    resources = "l8cloudmasks.zip"
    
    def __init__(self, root: str, data_mode: str = 'train'):
        self.root = root
        self.data_mode = data_mode

        # if not self._check_exists():
        #     self.download()
        #     self.extract_file()

        self.coords, self.chips, self.classes = \
            self.get_path_and_label()
        
        if self.data_mode == 'train':
            self.imgs = list(os.listdir(os.path.join(self.root, 'xview/train_images')))
        elif self.data_mode == 'validation':
            self.imgs = list(os.listdir(os.path.join(self.root, 'xview/val_images')))
        else:
            raise ValueError("data_mode not recognized. Try 'train' or 'validation'")

    def _check_exists(self) -> None:
        """TODO"""
        return

    def download(self):
       """Download file"""
       file_url = posixpath.join(self.mirrors, self.resources)
       _urlretrieve(file_url, os.path.join(self.root, self.resources))

    def extract_file(self):
        """Extract the .zip file"""
        shutil.unpack_archive(os.path.join(self.root, self.resources), self.root)
        os.remove(os.path.join(self.root, self.resources))

    def _check_exists_label(self, filename):
        """Check whether bounding boxes, image filenames, and labels 
        are already extracted from xView_train.geojson
        """
        path_to_check = os.path.join(self.root, 'xview', filename)
        return path_to_check, os.path.exists(path_to_check)

    def get_path_and_label(self):
        """Gets bounding boxes, image filenames, and labels 
        from xView_train.geojson
        
        Returns:
            coords: coordinates of the bounding boxes
            chips: image file names
            classes: classes for each ground truth
        """
        # check existance
        coords_path, coords_exists = self._check_exists_label('coords.npy')
        chips_path, chips_exists = self._check_exists_label('chips.npy')
        classes_path, classes_exists = self._check_exists_label('classes.npy')
        
        # if exist, load and return
        if coords_exists and chips_exists and classes_exists:
            coords = np.load(coords_path)
            chips = np.load(chips_path)
            classes = np.load(classes_path)
            return coords, chips, classes
        
        # read xView_train.geojson
        fname = os.path.join(self.root, 'xview/xView_train.geojson')
        with open(fname) as f:
            data = json.load(f)

        # initialize
        coords = []
        chips = []
        classes = []

        # extract
        feat_len = len(data['features'])
        for i in range(feat_len):
            properties = data['features'][i]['properties']
            b_id = properties['image_id']
            val = [int(num) for num in properties['bounds_imcoords'].split(',')]

            # type_id 75 and 82 don't belong to any class
            # https://github.com/DIUx-xView/xView1_baseline/issues/3
            if properties['type_id'] not in [75, 82]:
                chips.append(b_id)
                classes.append(properties['type_id'])
                coords.append(val)
        
        # convert to numpy arrays and save
        coords = np.array(coords)
        chips = np.array(chips)
        classes = np.array(classes)
        np.save(coords_path, coords)
        np.save(chips_path, chips)
        np.save(classes_path, classes)
        
        return coords, chips, classes
    
    def __getitem__(self, idx):
        """Returns a tensor image and a dictionary of target 
        consists of bounding boxes and labels
        """
        if self.data_mode == 'train':
            # image
            img_path = os.path.join(self.root, 'xview/train_images', self.imgs[idx])
            image = _load_img(img_path)
            image = np.array(image)
            image = torch.from_numpy(image)

            # bounding box
            bbox = self.coords[self.chips == self.imgs[idx]]
            bbox = torch.from_numpy(bbox)

            # label
            label = self.classes[self.chips == self.imgs[idx]]
            label = torch.from_numpy(label)
            
            # combine bounding box and label
            target = {}
            target['boxes'] = bbox
            target['labels'] = label

            sample = (image, target)
        elif self.data_mode == 'validation':
            # image
            img_path = os.path.join(self.root, 'xview/val_images', self.imgs[idx])
            image = _load_img(img_path)
            image = np.array(image)
            image = torch.from_numpy(image)

            sample = image
        else:
            raise ValueError("data_mode not recognized. Try 'train' or 'validation'")
        
        return sample


    def __len__(self):
        return len(self.imgs)
        

    def __iter__(self):
        for index in range(self.__len__()):
            yield self.__getitem__(index)