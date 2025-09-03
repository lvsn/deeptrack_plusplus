import torch.nn as nn
import torch


class Fire(nn.Module):
    """
    From SqueezeNet : https://github.com/pytorch/vision/blob/master/torchvision/models/squeezenet.py

    """
    def __init__(self, inplanes, squeeze_planes, expand1x1_planes, expand3x3_planes, stride=1, skip_last_activation=False):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.skip_last_activation = skip_last_activation
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ELU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1, stride=stride)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1, stride=stride)
        self.expanded_activation = nn.ELU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        x = torch.cat([self.expand1x1(x), self.expand3x3(x)], 1)
        if not self.skip_last_activation:
            x = self.expanded_activation(x)
        return x