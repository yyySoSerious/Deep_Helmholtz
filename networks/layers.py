import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, stereo_type='mvs',
                relu=True, bn=True, bn_momentum=0.1, **kwargs):
        super(Conv2dLayer, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, bias=(not bn), **kwargs)
        self.kernel_size = kernel_size
        self.stride = stride
        self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu
        self.stereo_type = stereo_type

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.stereo_type == 'mvs' and self.relu:
            x = F.relu_(x)
        elif self.stereo_type == 'ps':
            x = F.leaky_relu_(x, 0.1)

        return x


class Deconv2dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, stereo_type='mvs',
                relu=True, bn=True, bn_momentum=0.1, **kwargs):
        super(Deconv2dLayer, self).__init__()
        self.out_channels = out_channels
        assert stride in [1, 2]
        self.stride = stride

        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu
        self.stereo_type = stereo_type

    def forward(self, x):
        y = self.conv(x)
        if self.stereo_type == "mvs":
            if self.stride == 2:
                h, w = list(x.size())[2:]
                y = y[:, :, :2 * h, :2 * w].contiguous()
            if self.bn is not None:
                x = self.bn(y)
            if self.relu:
                x = F.relu_(x)
        elif self.stereo_type == "ps":
            x = F.leaky_relu_(y, 0.1)

        return x


class Conv3dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, relu=True, bn=True, bn_momentum=0.1,
                init_method='xavier', **kwargs):
        super(Conv3dLayer, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        assert stride in [1, 2]
        self.stride = stride

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm3d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu:
            x = F.relu_(x)

        return x

class Deconv3dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, relu=True, bn=True, bn_momentum=0.1,
                 init_method='xavier', **kwargs):
        super(Deconv3dLayer, self).__init__()
        self.out_channels = out_channels
        assert stride in [1, 2]
        self.stride = stride

        self.conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm3d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

    def forward(self, x):
        y = self.conv(x)
        if self.bn is not None:
            x = self.bn(y)
        if self.relu:
            x = F.relu_(x)

        return x


class Deconv2dLayerPlus(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, relu=True, bn=True, bn_momnetum=0.1,
                 add_type='channel_append'):
        super(Deconv2dLayerPlus, self).__init__()

        self.deconv = Deconv2dLayer(in_channels, out_channels, kernel_size, stride=2, padding=1, output_padding=1,
                                    bn=True, relu=relu, bn_momentum=bn_momnetum)
        self.add_type = add_type

    def forward(self, x_pre, x):
        x = self.deconv(x)
        x = x + x_pre if self.add_type == 'element_wise' else torch.cat((x, x_pre), dim=1)

        return x