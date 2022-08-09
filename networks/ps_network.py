import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_
from collections import OrderedDict

from . import layers
from .net_functions import *


def init_layers(num_layers, in_ch, out_ch, kernel_size, stride=1, bn=True):
    group_layers = OrderedDict()
    for i in range(num_layers):
        input_channels = in_ch if i == 0 else out_ch
        group_layers[str(i)] = layers.Conv2dLayer(input_channels, out_ch, kernel_size, stride,
                                                  stereo_type='ps', bn=bn, padding=1)

    return group_layers

class FeatExtractorNet(nn.Module):
    def __init__(self, in_channels, base_channels=64, bn=False, add_type='channel_append', nn_layers=(1, 1, 1, 1, 2, 1)):
        super(FeatExtractorNet, self).__init__()


        self.conv0 = nn.Sequential(init_layers(nn_layers[0], in_channels, base_channels, 3, 1, bn)
            #layers.Conv2dLayer(in_channels, base_channels, 3, 1, stereo_type='ps', bn=bn, padding=1),
        )

        self.conv1 = nn.Sequential(
            layers.Conv2dLayer(base_channels, base_channels * 2, 3, 2, stereo_type='ps', bn=bn, padding=1),
            nn.Sequential(init_layers(nn_layers[1], base_channels * 2, base_channels * 2, 3, 1, bn))
           # layers.Conv2dLayer(base_channels * 2, base_channels * 2, 3, 1, stereo_type='ps', bn=bn, padding=1)
        )

        self.conv2 = nn.Sequential(
            layers.Conv2dLayer(base_channels * 2, base_channels * 4, 3, 2, stereo_type='ps', bn=bn, padding=1),
            nn.Sequential(init_layers(nn_layers[2], base_channels * 4, base_channels * 4, 3, 1, bn))
            #layers.Conv2dLayer(base_channels * 4, base_channels * 4, 3, 1, stereo_type='ps', bn=bn, padding=1)
        )

        self.deconv = layers.Deconv2dLayerPlus(base_channels * 4, base_channels * 2, 3, stereo_type='ps',
                                               add_type=add_type)

        channels = base_channels * 2 if (add_type == 'element_wise') else base_channels*4

        self.conv3 = nn.Sequential(init_layers(nn_layers[3], channels, base_channels * 2, 3, 1, bn)
            #layers.Conv2dLayer(channels, base_channels * 2, 3, 1, stereo_type='ps', bn=bn,
                                     #   padding=1)
        )


    def forward(self, x):
        out1 = self.conv0(x)
        conv1 = self.conv1(out1)
        conv2 = self.conv2(conv1)
        deconv = self.deconv(conv1, conv2)
        out2 = self.conv3(deconv)

        b, c, h, w = out2.data.shape
        out2 = out2.view(-1)

        return out1, out2, [b, c, h, w]


class RegressionNet(nn.Module):
    def __init__(self, base_channels, bn=False, add_type='channel_append', nn_layers=(1, 1, 1, 1, 2, 1)):
        super(RegressionNet, self).__init__()

        self.conv0 = nn.Sequential(init_layers(nn_layers[0], base_channels, base_channels, 3, 1, bn)
            #layers.Conv2dLayer(base_channels, base_channels, 3, 1, stereo_type='ps', bn=bn, padding=1),
            #layers.Conv2dLayer(base_channels, base_channels, 3, 1, stereo_type='ps', bn=bn, padding=1)
        )
        self.deconv = layers.Deconv2dLayerPlus(base_channels, base_channels//2, 3, stereo_type='ps', add_type=add_type)
        channels = base_channels//2 if (add_type == 'element_wise') else base_channels

        self.conv1 = nn.Sequential(nn.Sequential(init_layers(nn_layers[1], channels, channels, 3, 1, bn)),
            nn.Conv2d(channels, 3, 3, 1, bias=False, padding=1)
        )

    def forward(self, x, pre):
        conv0 = self.conv0(x)
        deconv = self.deconv(pre, conv0)

        normal = self.conv1(deconv)
        normal = torch.nn.functional.normalize(normal, 2, 1)

        return normal

class PsNet(nn.Module):
    '''
    add_type = 'channel_append' or 'element_wise' (combining layers)
    fuse_type= 'max' or 'mean'
    '''
    def __init__(self, in_channels=6, base_channels=64, fuse_type='max', rcpcl_fuse_type='max', bn=False,
                 add_type='channel_append', nn_layers=(1, 1, 1, 1, 2, 1)):
        super(PsNet, self).__init__()
        self.feature_extractor = FeatExtractorNet(in_channels=in_channels, base_channels=base_channels, bn=bn,
                                                  add_type=add_type, nn_layers=nn_layers[:4])
        self.regressor = RegressionNet(base_channels=base_channels*2, bn=True, add_type=add_type, nn_layers=nn_layers[4:])
        self.fuse_type = fuse_type
        self.rcpcl_fuse_type = rcpcl_fuse_type

        #for m in self.modules():
         #   if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
          #      kaiming_normal_(m.weight.data)
          #      if m.bias is not None:
          #          m.bias.data.zero_()
          #  elif isinstance(m, nn.BatchNorm2d):
          #      m.weight.data.fill_(1)
          #      m.bias.data.zero_()

    def pool_features(self, features, fuse_type):
        pooled_feat = None
        if fuse_type == 'mean':
            pooled_feat = torch.stack(features, 1).mean(1)
        elif fuse_type == 'max':
            pooled_feat, _ = torch.stack(features, 1).max(1)

        return pooled_feat

    def pool_rcpcl_features(self, feat1, feat2, proj1, proj2, shape):
        feat2 = warp(feat2, proj2, proj1)
        if feat1.dim() >1:
            feat1 = feat1.view(-1)

        reciprocal_feats = [feat1, feat2.view(-1)]
        pooled_reciprocal_feat = self.pool_features(reciprocal_feats, self.rcpcl_fuse_type)
        pooled_reciprocal_feat = pooled_reciprocal_feat.view(shape[0], shape[1], shape[2], shape[3])

        return pooled_reciprocal_feat

    def forward(self, images, projection_mats):
        '''

        :param images: shape:- (B, num_src_images + 1, 6, H, W)
        :param projection_mats: shape:- (B, num_src_images + 1, 2, 4, 4)
        :return:
        '''
        features = []
        lost_features =[]
        shape = None
        lost_shape = None
        for idx in range(images.shape[1]):
            img = images[:, idx]
            lost_feat, feat, shape = self.feature_extractor(img)
            features.append(feat)
            lost_features.append(lost_feat)
            lost_shape = lost_feat.shape

        ref_feat = features[0]
        feats = [ref_feat.view(-1)]
        ref_lost_feat = lost_features[0]
        lost_feats = [ref_lost_feat.view(-1)]
        projection_mats = torch.unbind(projection_mats, 1)

        for i in range(0, len(features) - 1, 2):
            ref_proj = projection_mats[0]
            rcpcl1_proj, rcpcl2_proj = projection_mats[i + 1], projection_mats[i + 2]

            rcpcl2_proj_new = rcpcl2_proj[:, 0].clone()
            rcpcl2_proj_new[:, :3, :4] = torch.matmul(rcpcl2_proj[:, 1, :3, :3], rcpcl2_proj[:, 0, :3])
            rcpcl1_proj_new = rcpcl1_proj[:, 0].clone()
            rcpcl1_proj_new[:, :3, :4] = torch.matmul(rcpcl1_proj[:, 1, :3, :3], rcpcl1_proj[:, 0, :3, :4])

            rcpcl1_feat = features[i + 1]
            rcpcl2_feat = features[i + 2].view(shape[0], shape[1], shape[2], shape[3])
            pooled_reciprocal_feat = self.pool_rcpcl_features(rcpcl1_feat, rcpcl2_feat, rcpcl1_proj_new,
                                                              rcpcl2_proj_new, shape)

            rcpcl1_lost_feat = lost_features[i + 1]
            rcpcl2_lost_feat = lost_features[i + 2]
            pooled_reciprocal_lost_feat = self.pool_rcpcl_features(rcpcl1_lost_feat, rcpcl2_lost_feat, rcpcl1_proj_new,
                                                                   rcpcl2_proj_new, lost_shape)

            rcpcl1_proj_new[:, :3, :4] = torch.matmul(rcpcl1_proj[:, 1, :3, :3], rcpcl1_proj[:, 0, :3])
            ref_proj_new = ref_proj[:, 0].clone()
            ref_proj_new[:, :3, :4] = torch.matmul(ref_proj[:, 1, :3, :3], ref_proj[:, 0, :3, :4])
            pooled_reciprocal_feat = warp(pooled_reciprocal_feat, rcpcl1_proj_new, ref_proj_new)
            pooled_reciprocal_lost_feat = warp(pooled_reciprocal_lost_feat, rcpcl1_proj_new, ref_proj_new)

            feats.append(pooled_reciprocal_feat.view(-1))
            lost_feats.append(pooled_reciprocal_lost_feat.view(-1))

        pooled_feat = self.pool_features(feats, self.fuse_type).view(shape[0], shape[1], shape[2], shape[3])
        pooled_lost_feat = self.pool_features(lost_feats, self.fuse_type).view(lost_shape[0], lost_shape[1], lost_shape[2],
                                                               lost_shape[3])
        normal = self.regressor(pooled_feat, pooled_lost_feat)

        return normal