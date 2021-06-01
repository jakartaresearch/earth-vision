class LandCover():

    """
    The LandCover.ai (Land Cover from Aerial Imagery) dataset.
    <https://landcover.ai/download/landcover.ai.v1.zip>
    """

    mirrors = "https://landcover.ai/download/"
    resources = "landcover.ai.v1.zip"

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


  
