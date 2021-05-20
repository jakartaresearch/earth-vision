"""RESISC45 Dataset."""


class RESISC45():
    def __init__(self):
        raise NotImplementedError

    def __itemget__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError

    def download(self):
        raise NotImplementedError

    def _check_exists(self):
        raise NotImplementedError
