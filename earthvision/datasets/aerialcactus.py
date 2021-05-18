class AerialCactus():
    """Aerial Cactus Dataset.
    <https://www.kaggle.com/c/aerial-cactus-identification>
    """

    def __init__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def download(self):
        """download and extract file.
        """
        raise NotImplementedError
