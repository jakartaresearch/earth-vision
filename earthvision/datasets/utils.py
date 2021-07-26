import sys
import os
import urllib
import numpy as np
import boto3

from tqdm import tqdm
from PIL import Image
from botocore import UNSIGNED
from botocore.client import Config


def _urlretrieve(url: str, filename: str, chunk_size: int = 1024) -> None:
    with open(filename, "wb") as fh:
        with urllib.request.urlopen(urllib.request.Request(url)) as response:
            with tqdm(total=response.length) as pbar:
                for chunk in iter(lambda: response.read(chunk_size), ""):
                    if not chunk:
                        break
                    pbar.update(chunk_size)
                    fh.write(chunk)


def s3_downloader(s3_client, local_file_name, s3_bucket, s3_object_key):
    """Download dataset from Amazon S3.

    Args:
        s3_client: Object boto3.client.
        local_file_name: Destination filepath.
        s3_bucket: S3 bucket.
        s3_object_key: S3 object key.

    """
    meta_data = s3_client.head_object(Bucket=s3_bucket, Key=s3_object_key)
    total_length = int(meta_data.get('ContentLength', 0))
    downloaded = 0

    def progress(chunk):
        nonlocal downloaded
        downloaded += chunk
        done = int(50 * downloaded / total_length)
        sys.stdout.write("\r[%s%s]" % ('=' * done, ' ' * (50-done)))
        sys.stdout.flush()

    print(f'Downloading {s3_object_key}')
    with open(local_file_name, 'wb') as f:
        s3_client.download_fileobj(
            s3_bucket, s3_object_key, f, Callback=progress)


def downloader(resource, root):
    """Downloader function that handle general download link or S3 cloud storage.

    Args:
        resource: Dataset resource link.
        root: Dataset destination filepath.

    """
    resource_type, obj = resource.split('://')[0], resource.split('://')[1]
    dest_pth = os.path.join(root, resource.split('/')[-1])

    if resource_type == 's3':
        s3_client = boto3.client(
            's3', config=Config(signature_version=UNSIGNED))
        bucket = obj.split('/')[0]
        obj_key = '/'.join(obj.split('/')[1:])
        s3_downloader(s3_client, dest_pth, bucket, obj_key)
    else:
        _urlretrieve(resource, dest_pth)


def _load_img(fname):
    return Image.open(fname)


def _load_npy(fname):
    return np.load(fname)
