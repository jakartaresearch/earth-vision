"""Aerial Cactus Dataset"""


class AerialCactus():
    """Aerial Cactus <https://www.kaggle.com/c/aerial-cactus-identification> Dataset. 
    """

    def __init__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def download(self):
        raise NotImplementedError
