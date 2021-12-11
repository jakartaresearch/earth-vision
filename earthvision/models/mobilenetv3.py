"""Inspired by torchvision.models.mobilenetv3"""

from torch import nn
from typing import Any, Callable, List, Optional
from .utils import load_state_dict_from_url
from torchvision.models.mobilenetv3 import MobileNetV3, InvertedResidualConfig, _mobilenet_v3_conf

__all__ = ["MobileNetV3", "mobilenet_v3_large"]


model_urls = {
    "mobilenet_v3_large": (
        "https://drive.google.com/uc?id=1--_vx4lTMSKmW1X3DS1KXcewXdmBMu-K",
        "resisc45_mobilenetv3_large.pth"
    )
}

class OurMobileNetV3(MobileNetV3):
    def __init__(self,
            inverted_residual_setting: List[InvertedResidualConfig],
            last_channel: int,
            num_classes: int = 45,
            block: Optional[Callable[..., nn.Module]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            **kwargs: Any
            ) -> None:
        super().__init__(inverted_residual_setting, last_channel, num_classes=num_classes, block=block, norm_layer=norm_layer, **kwargs)

def _mobilenet_v3_model(
    arch: str,
    inverted_residual_setting: List[InvertedResidualConfig],
    last_channel: int,
    pretrained: bool,
    **kwargs: Any
):
    model = OurMobileNetV3(inverted_residual_setting, last_channel, **kwargs)
    if pretrained:
        if model_urls.get(arch, None) is None:
            raise ValueError("No checkpoint is available for model type {}".format(arch))
        state_dict = load_state_dict_from_url(model_urls[arch])
        model.load_state_dict(state_dict)
    return model

def mobilenet_v3_large(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> MobileNetV3:
    """
    Constructs a large MobileNetV3 architecture from
    `"Searching for MobileNetV3" <https://arxiv.org/abs/1905.02244>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    arch = "mobilenet_v3_large"
    inverted_residual_setting, last_channel = _mobilenet_v3_conf(arch, **kwargs)
    return _mobilenet_v3_model(arch, inverted_residual_setting, last_channel, pretrained, **kwargs)

