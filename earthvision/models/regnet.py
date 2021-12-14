# Modified from
# https://github.com/facebookresearch/ClassyVision/blob/main/classy_vision/models/anynet.py
# https://github.com/facebookresearch/ClassyVision/blob/main/classy_vision/models/regnet.py
from functools import partial
from typing import Any

import torch
from torch import nn
from torchvision.models.regnet import BlockParams
from torchvision.models import RegNet

from .utils import load_state_dict_from_url


__all__ = ["RegNet", "regnet_y_400mf"]


model_urls = {
    "regnet_y_400mf": (
        "https://drive.google.com/uc?id=1gtoXOxQwt8_J64qFsYsXFh2iQPeln0bq",
        "resisc45_regnet_y_400mf.pth",
    )
}


class RegNet45Class(RegNet):
    def __init__(self, block_params, norm_layer):
        super().__init__(block_params, norm_layer=norm_layer, num_classes=45)


def _regnet(
    arch: str, block_params: BlockParams, pretrained: bool, progress: bool, **kwargs: Any
) -> RegNet45Class:
    norm_layer = kwargs.pop("norm_layer", partial(nn.BatchNorm2d, eps=1e-05, momentum=0.1))
    model = RegNet45Class(block_params, norm_layer=norm_layer, **kwargs)
    if pretrained:
        if arch not in model_urls:
            raise ValueError(f"No checkpoint is available for model type {arch}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dict = load_state_dict_from_url(model_urls[arch], map_location=device)
        model.load_state_dict(state_dict)
    return model


def regnet_y_400mf(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> RegNet:
    """
    Constructs a RegNetY_400MF architecture from
    `"Designing Network Design Spaces" <https://arxiv.org/abs/2003.13678>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    params = BlockParams.from_init_params(
        depth=16, w_0=48, w_a=27.89, w_m=2.09, group_width=8, se_ratio=0.25, **kwargs
    )
    return _regnet("regnet_y_400mf", params, pretrained, progress, **kwargs)
