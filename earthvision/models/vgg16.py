from torchvision.models import VGG

__all__ = ['VGG', 'vgg16']

model_urls = {
    "vgg16": (
        "https://drive.google.com/uc?id=1XW0iB-068A-knPXgZL3gSjvgDXjymy0Q",
        "resisc45_vgg16.pth",
    )
}
