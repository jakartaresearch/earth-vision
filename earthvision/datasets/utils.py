"""Utility functions."""
import sys
import os
import urllib
import collections
import ssl
import numpy as np
import boto3
from spectral import open_image

from tqdm import tqdm
from PIL import Image
from botocore import UNSIGNED
from botocore.client import Config

# define MAX_IMAGE_PIXELS
Image.MAX_IMAGE_PIXELS = 1000000000


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


def s3_downloader(s3_client, local_file_name: str, s3_bucket: str, s3_object_key: str):
    """Download dataset from Amazon S3.

    Args:
        s3_client: Object boto3.client.
        local_file_name: Destination filepath.
        s3_bucket: S3 bucket.
        s3_object_key: S3 object key.

    """
    meta_data = s3_client.head_object(Bucket=s3_bucket, Key=s3_object_key)
    total_length = int(meta_data.get("ContentLength", 0))
    downloaded = 0

    def progress(chunk):
        nonlocal downloaded
        downloaded += chunk
        done = int(50 * downloaded / total_length)
        sys.stdout.write("\r[%s%s]" % ("=" * done, " " * (50 - done)))
        sys.stdout.flush()

    print(f"Downloading {s3_object_key}")
    with open(local_file_name, "wb") as f:
        s3_client.download_fileobj(s3_bucket, s3_object_key, f, Callback=progress)


def downloader(resource: str, root: str):
    """Downloader function that handle general download link or S3 cloud storage.

    Args:
        resource: Dataset resource link.
        root: Dataset destination filepath.

    """
    resource_type, obj = resource.split("://")[0], resource.split("://")[1]
    dest_pth = os.path.join(root, resource.split("/")[-1])

    if resource_type == "s3":
        s3_client = boto3.client("s3", config=Config(signature_version=UNSIGNED))
        bucket = obj.split("/")[0]
        obj_key = "/".join(obj.split("/")[1:])
        s3_downloader(s3_client, dest_pth, bucket, obj_key)
    else:
        _urlretrieve(resource, dest_pth)


def _load_img(fname):
    return Image.open(fname)


def _load_npy(fname):
    return np.load(fname)


def _load_img_hdr(fname):
    return open_image(fname).read_band(0)


def _resize_stack(ls):
    ls_size = [im.size for im in ls]

    h, w = zip(*ls_size)

    h_mode = list(collections.Counter(h))[0]
    w_mode = list(collections.Counter(w))[0]

    for idx, (h, w) in enumerate(ls_size):
        if (h != h_mode) | ((w != w_mode)):
            ls[idx] = ls[idx].resize((h_mode, w_mode))

    return ls


def _load_stack_img(list_path_file):
    ls = [Image.open(file_name) for file_name in list_path_file]

    ls = _resize_stack(ls)
    stack_img = np.stack(ls)
    stack_img = stack_img.astype(np.int16)
    return stack_img
