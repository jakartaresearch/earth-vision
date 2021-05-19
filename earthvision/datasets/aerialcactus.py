"""Aerial Cactus Dataset from Kaggle."""
import os
import posixpath
import shutil
from .utils import _urlretrieve


class AerialCactus():
    """Aerial Cactus Dataset.

    <https://www.kaggle.com/c/aerial-cactus-identification>
    """

    mirrors = "https://storage.googleapis.com/ossjr"
    resources = "cactus-aerial-photos.zip"

    def __init__(self, root):
        self.root = root
        if not self._check_exists():
            self.download()
            self.extract_file()

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

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
