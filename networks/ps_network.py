import torch.nn as nn
from . import layers

class FeatExtractorNet(nn.Module):
    def __init__(self, base_channels):
        super(FeatExtractorNet, self).__init__()
        pass

    def forward(self, x):
        pass


class RegressionNet(nn.Module):
    def __init__(self, base_channels, bn=False):
        super(RegressionNet, self).__init__()
        self.deconv1 = layers.Conv2dLayer(base_channels, base_channels, 3, 1, stereo_type='ps', bn=bn, padding=1)
        self.deconv2 = layers.Conv2dLayer(base_channels, base_channels, 3, 1, stereo_type='ps', bn=bn, padding=1)
        self.deconv3 = layers.Deconv2dLayer(base_channels, base_channels//2, 4, 2, stereo_type='ps', bn=bn, padding=1)
        self.conv1 = layers.Conv2dLayer(base_channels//2, 3, 3, 1, stereo_type='ps', bn=bn, padding=1)

    def forward(selfself, x, shape):
        pass