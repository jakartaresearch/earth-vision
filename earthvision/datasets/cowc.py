"""Cars Overhead with Context."""
import os
import shutil
import posixpath
import torch
from torch.utils.data import Dataset
from .utils import _urlretrieve, _load_img


class COWC():
    """ Cars Overhead with Context.
    https://gdo152.llnl.gov/cowc/
    """

    mirrors = "https://storage.googleapis.com/ossjr"
    resources = "cowc-everything-small.txz"

    def __init__(self,
                 root: str,
                 data_mode: str = 'train',
                 transform=None,
                 target_transform=None):
        self.root = root
        self.data_mode = data_mode
        self.transform = transform
        self.target_transform = target_transform

        if not self._check_exists():
            self.download()
            self.extract_file()

        # self.img_labels = self.get_path_and_label()

    def __getitem__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, "cowc"))

    def download(self):
        """download file."""
        file_url = posixpath.join(self.mirrors, self.resources)
        _urlretrieve(file_url, self.resources)

    def extract_file(self):
        """Extract file from compressed."""
        shutil.unpack_archive(self.resources, self.root)
        os.remove(os.path.join(self.root, self.resources))
