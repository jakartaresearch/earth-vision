from torch import nn
from typing import Any
from torchvision.models import VGG
from .utils import load_state_dict_from_url
from torchvision.models.vgg import make_layers, cfgs

__all__ = ['VGG', 'vgg16']

model_urls = {
    "vgg16": (
        "https://drive.google.com/uc?id=1XW0iB-068A-knPXgZL3gSjvgDXjymy0Q",
        "resisc45_vgg16.pth",
    )
}

class VGG16(VGG):
    def __init__(
        self,
        features: nn.Module,
        num_classes: int = 45,
        init_weights: bool = True
    ) -> None:
        super().__init__(features, num_classes, init_weights)

def _vgg(arch: str, cfg: str, batch_norm: bool, pretrained: bool, progress: bool, **kwargs: Any) -> VGG:
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model

def vgg16(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG16:
    r"""VGG 16-layer model (configuration "D")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    The required minimum input size of the model is 32x32.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16', 'D', False, pretrained, progress, **kwargs)