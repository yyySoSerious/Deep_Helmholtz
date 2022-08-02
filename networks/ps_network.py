import torch
import torch.nn as nn
from . import layers

class FeatExtractorNet(nn.Module):
    def __init__(self, base_channels, bn=False):
        super(FeatExtractorNet, self).__init__()
        self.deconv = layers.Deconv2dLayer(base_channels, base_channels//2, 4, 2, stereo_type='ps', bn=bn, padding=1)
        self.conv = layers.Conv2dLayer(base_channels//2, base_channels//2, 3, 1, stereo_type='ps', bn=bn, padding=1)

    def forward(self, x):
        deconv = self.deconv(x)
        out = self.conv(deconv)
        b, c, h, w = out.data.shape
        out = out.view(-1)
        return out, [b, c, h, w]


class UNet(nn.Module):
    def __init__(self, in_channels, base_channels, bn=False):
        super(UNet, self).__init__()

        self.conv0 = layers.Conv2dLayer(in_channels, base_channels, 3, 1, stereo_type='mvs', bn=bn, padding=1)

        self.conv1 = layers.Conv2dLayer(base_channels, base_channels * 2, 3,  stride=2,  stereo_type='mvs', bn=bn, padding=1)
        self.conv2 = layers.Conv2dLayer(base_channels * 2, base_channels * 2,  3, 1,  stereo_type='mvs', bn=bn, padding=1)

        self.conv3 = layers.Conv2dLayer(base_channels * 2, base_channels * 4, 3, stride=2,  stereo_type='mvs', bn=bn, padding=1)
        self.conv4 = layers.Conv2dLayer(base_channels * 4, base_channels * 4,3, 1,  stereo_type='mvs', bn=bn, padding =1)

        self.conv5 = layers.Conv2dLayer(base_channels * 4, base_channels * 8, 3, stride=2,  stereo_type='mvs', bn=bn, padding=1)
        self.conv6 = layers.Conv2dLayer(base_channels * 8, base_channels * 8, 3, 1,  stereo_type='mvs', bn=bn, padding=1)

        self.deconv7 = layers.Deconv2dLayer(base_channels * 8, base_channels * 4, 3, stride=2,  stereo_type='mvs', bn=bn, padding=1,
                                            output_padding=1)

        self.deconv8 = layers.Deconv2dLayer(base_channels * 4, base_channels * 2, 3, stride=2,  stereo_type='mvs', bn=bn, padding=1,
                                            output_padding=1)

        self.deconv8 = layers.Deconv2dLayer(base_channels * 4, base_channels * 2, 3, stride=2,  stereo_type='mvs', bn=bn, padding=1,
                                            output_padding=1)

        self.deconv9 = layers.Deconv2dLayer(base_channels * 2, base_channels * 1, 3, stride=2,  stereo_type='mvs', bn=bn, padding=1,
                                            output_padding=1)

        self.conv7 = layers.Conv2dLayer(base_channels, 3, 3, 1, stereo_type='mvs', bn=bn, padding=1)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        x = self.conv6(self.conv5(conv4))
        x = conv4 + self.deconv7(x)
        x = conv2 + self.deconv8(x)
        x = conv0 + self.deconv9(x)
        normal = self.conv7(x)

        normal = torch.nn.functional.normalize(normal, 2, 1)
        return normal


class RegressionNet(nn.Module):
    def __init__(self, base_channels, bn=False):
        super(RegressionNet, self).__init__()
        self.conv1 = layers.Conv2dLayer(base_channels, base_channels, 3, 1, stereo_type='ps', bn=bn, padding=1)
        self.conv2 = layers.Conv2dLayer(base_channels, base_channels, 3, 1, stereo_type='ps', bn=bn, padding=1)
        #self.deconv = layers.Deconv2dLayer(base_channels, base_channels//2, 4, 2, stereo_type='ps', bn=bn, padding=1)
        self.conv3 = layers.Conv2dLayer(base_channels, 3, 3, 1, stereo_type='ps', bn=bn, padding=1)#layers.Conv2dLayer(base_channels//2, 3, 3, 1, stereo_type='ps', bn=bn, padding=1)

    def forward(self, x, shape):
        x = x.view(shape[0], shape[1], shape[2], shape[3])
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        #test = self.conv2(conv1)
        #deconv = self.deconv(conv2)
        normal = self.conv3(conv2) # self.conv3(deconv)
        normal = torch.nn.functional.normalize(normal, 2, 1)
        return normal