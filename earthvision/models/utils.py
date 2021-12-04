import os
import warnings
import torch
import gdown
import collections.abc

ENV_TORCH_HOME = "TORCH_HOME"
ENV_XDG_CACHE_HOME = "XDG_CACHE_HOME"
DEFAULT_CACHE_DIR = "~/.cache"
_hub_dir = None


def _get_torch_home():
    torch_home = os.path.expanduser(
        os.getenv(
            ENV_TORCH_HOME, os.path.join(os.getenv(ENV_XDG_CACHE_HOME, DEFAULT_CACHE_DIR), "torch")
        )
    )
    return torch_home


def get_dir():
    r"""
    Get the Torch Hub cache directory used for storing downloaded models & weights.
    If :func:`~torch.hub.set_dir` is not called, default path is ``$TORCH_HOME/hub`` where
    environment variable ``$TORCH_HOME`` defaults to ``$XDG_CACHE_HOME/torch``.
    ``$XDG_CACHE_HOME`` follows the X Design Group specification of the Linux
    filesystem layout, with a default value ``~/.cache`` if the environment
    variable is not set.
    """
    # Issue warning to move data if old env is set
    if os.getenv("TORCH_HUB"):
        warnings.warn("TORCH_HUB is deprecated, please use env TORCH_HOME instead")

    if _hub_dir is not None:
        return _hub_dir
    return os.path.join(_get_torch_home(), "hub")


def set_dir(d):
    r"""
    Optionally set the Torch Hub directory used to save downloaded models & weights.
    Args:
        d (string): path to a local folder to save downloaded models & weights.
    """
    global _hub_dir
    _hub_dir = d


def load_state_dict_from_url(url, model_dir=None, map_location=None):
    r"""Loads the Torch serialized object at the given URL.
    If downloaded file is a zip file, it will be automatically
    decompressed.
    If the object is already present in `model_dir`, it's deserialized and
    returned.
    The default value of ``model_dir`` is ``<hub_dir>/checkpoints`` where
    ``hub_dir`` is the directory returned by :func:`~torch.hub.get_dir`.
    Args:
        url (string): URL of the object to download
        model_dir (string, optional): directory in which to save the object
        map_location (optional): a function or a dict specifying how to remap storage locations
    """
    if model_dir is None:
        hub_dir = get_dir()
        model_dir = os.path.join(hub_dir, "checkpoints")

    os.makedirs(model_dir, exist_ok=True)
    cached_file = os.path.join(model_dir, url[1])
    if not os.path.exists(cached_file):
        gdown.download(url[0], cached_file, quiet=False)
    return torch.load(cached_file, map_location=map_location)

# Inspired by
# https://github.com/rwightman/pytorch-image-models/blob/b9bd960a032c75ca6b808ddeed76bee5f3ed4972/timm/models/layers/helpers.py
# From PyTorch internals
def to_2tuple(x):
    if isinstance(x, collections.abc.Iterable):
        return x
    return (x, x)
