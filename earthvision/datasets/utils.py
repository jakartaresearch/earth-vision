import urllib
import ssl
import numpy as np
from tqdm import tqdm
from PIL import Image


def _urlretrieve(url: str, filename: str, chunk_size: int = 1024) -> None:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    with open(filename, "wb") as fh:
        with urllib.request.urlopen(urllib.request.Request(url), context=ctx) as response:
            with tqdm(total=response.length) as pbar:
                for chunk in iter(lambda: response.read(chunk_size), ""):
                    if not chunk:
                        break
                    pbar.update(chunk_size)
                    fh.write(chunk)


def _load_img(fname):
    return Image.open(fname)