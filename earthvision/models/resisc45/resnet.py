"""Inspired by torchvision.models.resnet"""
from typing import Type, Any, Union, List
from torchvision.models.resnet import BasicBlock, Bottleneck
from torchvision.models import ResNet

from .utils import load_state_dict_from_url

__all__ = ["ResNet", "resnet50"]


model_urls = {
    "resnet50": (
        "https://drive.google.com/uc?id=1TvwMlCPhaN6BlUk-DotjfcvMBZcsd2Vb",
        "resisc45_resnet50.pth",
    )
}


class ResNet45Class(ResNet):
    def __init__(self, block, layers):
        super().__init__(block, layers, num_classes=45)


def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any,
) -> ResNet45Class:
    model = ResNet45Class(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch])
        model.load_state_dict(state_dict)
    return model


def resnet50(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet50", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)
