from torch import nn as nn
import torchvision.models.resnet as resnet
from torchvision.models import ResNet
from .utils import load_state_dict_from_url

class BasicBlock_ext(resnet.BasicBlock):
    pass

class ResNetExtended(ResNet):
    def __init__(self, *args, **kwargs):
      
      super(ResNetExtended, self).__init__(*args, **kwargs)
      if hasattr(self, 'avgpool'):
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
      elif hasattr(self, 'features'):
        self.features.add_module('avgpool', nn.AdaptiveAvgPool2d((7, 7)))
      else:
        raise NotImplementedError

## The fowllowing methods were taken from the
## original repo: 
## https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNetExtended(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet18_ext(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock_ext, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


