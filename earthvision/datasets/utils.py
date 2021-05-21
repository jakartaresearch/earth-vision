import urllib
import numpy as np
from tqdm import tqdm
import torch
from PIL import Image


def _urlretrieve(url: str, filename: str, chunk_size: int = 1024) -> None:
    with open(filename, "wb") as fh:
        with urllib.request.urlopen(urllib.request.Request(url)) as response:
            with tqdm(total=response.length) as pbar:
                for chunk in iter(lambda: response.read(chunk_size), ""):
                    if not chunk:
                        break
                    pbar.update(chunk_size)
                    fh.write(chunk)

def _to_categorical(y, num_classes):
    """One-hot encode label y into a tensor with size (len(y), num_classes)"""
    y = np.array(y)
    y = torch.from_numpy(y)
    y = torch.nn.functional.one_hot(y, num_classes=num_classes)
    return y

def _load_img(fname):
    return Image.open(fname)