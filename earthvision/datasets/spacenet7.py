"""SpaceNet 7 Dataset: Multi-Temporal Urban Development Challenge - Instance Segmentation."""
from PIL import Image
import os
import shutil
import numpy as np
import pandas as pd
import multiprocessing

from typing import Any, Callable, Optional, Tuple
from .utils import downloader, _load_img
from .vision import VisionDataset
from .spacenet7_utils import map_wrapper, make_geojsons_and_masks


class SpaceNet7(VisionDataset):
    """SpaceNet7 (SN7): Multi-Temporal Urban Development Challenge
    
    <https://spacenet.ai/sn7-challenge/>

    Args:
        root (string): Root directory of dataset.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image and
            returns a transformed version. E.g, transforms.RandomCrop
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    resources = {
        "train": "s3://spacenet-dataset/spacenet/SN7_buildings/tarballs/SN7_buildings_train.tar.gz",
        "test": "s3://spacenet-dataset/spacenet/SN7_buildings/tarballs/SN7_buildings_test_public.tar.gz",
    }

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:

        super(SpaceNet7, self).__init__(
            root, transform=transform, target_transform=target_transform
        )

        self.root = root
        self.data_mode = "train" if train else "test"
        self.filename = self.resources.get(self.data_mode, "NULL").split("/")[-1]
        self.dataset_path = os.path.join(root, self.filename)
        data_mode_folder = {"train": "train", "test": "test_public"}
        self.folder_name = data_mode_folder.get(self.data_mode, "NULL")

        if not os.path.exists(self.root):
            os.makedirs(self.root)

        if download and self._check_exists(self.dataset_path):
            print("file already exists.")

        if download and not self._check_exists(os.path.join(self.root, self.folder_name)):
            self.download()
            self.extract_file()

        if self.data_mode == "train":
            aois = sorted(
                [
                    f
                    for f in os.listdir(os.path.join(self.root, "train"))
                    if os.path.isdir(os.path.join(self.root, "train", f))
                ]
            )

            aois_without_mask = []
            for aoi in aois:
                mask_dir = os.path.join(self.root, "train", aoi, "masks/")
                if not self._check_exists(mask_dir):
                    aois_without_mask.append(aoi)

            if aois_without_mask:
                print("Generating masks...")
                self.generate_mask(aois_without_mask)

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

    def generate_mask(self, aois):
        """
        Create Training Masks
        Multi-thread to increase speed
        We'll only make a 1-channel mask for now, but Solaris supports a multi-channel mask as well, see
            https://github.com/CosmiQ/solaris/blob/master/docs/tutorials/notebooks/api_masks_tutorial.ipynb
        """
        make_fbc = False

        input_args = []
        for i, aoi in enumerate(aois):
            print(i, "aoi:", aoi)
            im_dir = os.path.join(self.root, "train", aoi, "images_masked/")
            json_dir = os.path.join(self.root, "train", aoi, "labels_match/")
            out_dir_mask = os.path.join(self.root, "train", aoi, "masks/")
            out_dir_mask_fbc = os.path.join(self.root, "train", aoi, "masks_fbc/")
            os.makedirs(out_dir_mask, exist_ok=True)
            if make_fbc:
                os.makedirs(out_dir_mask_fbc, exist_ok=True)

            json_files = sorted(
                [
                    f
                    for f in os.listdir(os.path.join(json_dir))
                    if f.endswith("Buildings.geojson") and os.path.exists(os.path.join(json_dir, f))
                ]
            )
            for j, f in enumerate(json_files):
                # print(i, j, f)
                name_root = f.split(".")[0]
                json_path = os.path.join(json_dir, f)
                image_path = (
                    os.path.join(im_dir, name_root + ".tif")
                    .replace("labels", "images")
                    .replace("_Buildings", "")
                )
                output_path_mask = os.path.join(out_dir_mask, name_root + ".tif")
                if make_fbc:
                    output_path_mask_fbc = os.path.join(out_dir_mask_fbc, name_root + ".tif")
                else:
                    output_path_mask_fbc = None

                if os.path.exists(output_path_mask):
                    continue
                else:
                    input_args.append(
                        [
                            make_geojsons_and_masks,
                            name_root,
                            image_path,
                            json_path,
                            output_path_mask,
                            output_path_mask_fbc,
                        ]
                    )

        p = multiprocessing.Pool(multiprocessing.cpu_count() - 1)
        out = p.map(map_wrapper, input_args)
        p.close()
        p.join()

    def get_path_and_label(self):
        """Return dataframe type consist of image path and corresponding label (for train data),
        or image path only (for test data)."""
        pops = ["train", "test_public"]

        for pop in pops:
            d = os.path.join(self.root, pop)
            im_list, mask_list = [], []
            subdirs = sorted([f for f in os.listdir(d) if os.path.isdir(os.path.join(d, f))])
            for subdir in subdirs:
                if pop == "train":
                    im_files = [
                        os.path.join(d, subdir, "images_masked", f)
                        for f in sorted(os.listdir(os.path.join(d, subdir, "images_masked")))
                        if f.endswith(".tif")
                        and os.path.exists(
                            os.path.join(d, subdir, "masks", f.split(".")[0] + "_Buildings.tif")
                        )
                    ]
                    mask_files = [
                        os.path.join(d, subdir, "masks", f.split(".")[0] + "_Buildings.tif")
                        for f in sorted(os.listdir(os.path.join(d, subdir, "images_masked")))
                        if f.endswith(".tif")
                        and os.path.exists(
                            os.path.join(d, subdir, "masks", f.split(".")[0] + "_Buildings.tif")
                        )
                    ]
                    im_list.extend(im_files)
                    mask_list.extend(mask_files)

                elif pop == "test_public":
                    im_files = [
                        os.path.join(d, subdir, "images_masked", f)
                        for f in sorted(os.listdir(os.path.join(d, subdir, "images_masked")))
                        if f.endswith(".tif")
                    ]
                    im_list.extend(im_files)

            if self.data_mode == "train":
                df = pd.DataFrame({"image": im_list, "label": mask_list})
            elif self.data_mode == "test":
                df = pd.DataFrame({"image": im_list})

            return df

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """
        Args:
            idx (int): Index
        Returns:
            tuple: (img, mask) or (img)
        """
        img_path = self.img_labels.iloc[idx, 0]
        img = np.array(_load_img(img_path))

        if self.transform is not None:
            img = Image.fromarray(img)
            img = self.transform(img)

        if self.data_mode == "train":
            mask_path = self.img_labels.iloc[idx, 1]
            mask = np.array(_load_img(mask_path))

            if self.target_transform is not None:
                mask = Image.fromarray(mask)
                mask = self.target_transform(mask)
            sample = (img, mask)

        elif self.data_mode == "test":
            sample = img

        return sample

    def __len__(self) -> int:
        return len(self.img_labels)
