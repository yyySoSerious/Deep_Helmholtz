import torch.nn as nn
from . import mvs_network as mvs
from . import ps_network as ps
from .net_functions import *


class HelmholtzNet(nn.Module):
    def __init__(self, planes_in_stages=(64, 32, 8), conf_lambda=1.5, grad_method='detach',
                 base_channels_per_stage=(8, 8, 8), net_type='mvs'):
        super(HelmholtzNet, self).__init__()

        self.planes_in_stages = planes_in_stages
        self.conf_lambda = conf_lambda
        self.grad_method = grad_method
        self.base_channels_per_stage = base_channels_per_stage
        self.net_type = net_type
        self.num_stages = len(planes_in_stages)
        self.scale_factors_inv = {'stage1': 4.0, 'stage2': 2.0, 'stage3': 1.0}

        self.mvs_feature_extractor = mvs.FeatExtractorNet(in_channels=3, base_channels=base_channels_per_stage[0],
                                                          num_stages=self.num_stages)

        if self.net_type == 'mvs' or self.net_type == 'helmholtz':
            self.mvs_cost_regularizer = nn.ModuleList(
                [mvs.CostRegularizerNet(in_channels=self.mvs_feature_extractor.out_channels[i],
                                        base_channels=self.base_channels_per_stage[i]) for i in range(self.num_stages)])

        if self.net_type == 'ps' or self.net_type == 'helmholtz':
            #self.unet = ps.UNet(in_channels= 8+3, base_channels=8, bn=True)
           # self.ps_feature_extractor = ps.FeatExtractorNet(base_channels=base_channels_per_stage[0]*4)
            self.ps_mininet = ps.miniNet(base_channels=base_channels_per_stage[0]+3, bn=True)
            self.ps_regressor = ps.RegressionNet(base_channels=base_channels_per_stage[0]+3, bn=True)#*2, bn=False)

    def mvs_forward(self, features, img, projection_mats, depth_values):
        mvs_outputs = {}
        depth, curr_depth, expected_variance = None, None, None
        for stage_num in range(self.num_stages):
            curr_stage_features = [feat[f'stage{stage_num + 1}'] for feat in features]
            curr_stage_projection_mats = projection_mats[f'stage{stage_num + 1}']
            scale_factor_inv = self.scale_factors_inv[f'stage{stage_num + 1}']
            curr_height = img.shape[2] // int(scale_factor_inv)
            curr_width = img.shape[3] // int(scale_factor_inv)

            # Not the first stage
            if depth is not None:
                if self.grad_method == 'detach':
                    curr_depth = depth.detach()
                    expected_variance = expected_variance.detach()
                else:
                    curr_depth = depth

                curr_depth = F.interpolate(curr_depth.unsqueeze(1), [curr_height, curr_width], mode='bilinear')
                expected_variance = F.interpolate(expected_variance.unsqueeze(1), [curr_height, curr_width],
                                                  mode='bilinear')

            # The first stage
            else:
                curr_depth = depth_values

            depth_range_samples = uncertainty_aware_samples(curr_depth=curr_depth,
                                                            expected_variance=expected_variance,
                                                            num_depth=self.planes_in_stages[stage_num],
                                                            dtype=img[0].dtype,
                                                            device=img[0].device,
                                                            shape=[img.shape[0], curr_height, curr_width])

            curr_stage_outputs = compute_depth(curr_stage_features, curr_stage_projection_mats,
                                               depth_samples=depth_range_samples,
                                               cost_reg=self.mvs_cost_regularizer[stage_num],
                                               conf_lambda=self.conf_lambda,
                                               is_training=self.training)

            depth = curr_stage_outputs['depth']
            expected_variance = curr_stage_outputs['variance']

            mvs_outputs[f'stage{stage_num + 1}'] = curr_stage_outputs

        return mvs_outputs

    def ps_forward(self, features, projection_mats, lights): #stage3 proj_mat
        '''

        :param features: shape: (B, C, H, W)
        :param projection_mats: shape = (B, num_src_images + 1, 2, 4, 4)
        :param lights: shape =  (B, num_src_images + 1, 3, H, W)
        :return:
        '''
        ref_feat = torch.cat((features[0]['stage3'], lights[:, 0, ]), 1)
        feats = [ref_feat.view(-1)]
        curr_stage_projection_mats = torch.unbind(projection_mats, 1)
        shape = ref_feat.shape

        for i in range(0, len(features) - 1, 2):
            index = i + 1
            ref_proj = curr_stage_projection_mats[0]
            rcpcl1_proj, rcpcl2_proj = curr_stage_projection_mats[i+1], curr_stage_projection_mats[i+2]

            rcpcl2_proj_new = rcpcl2_proj[:, 0].clone()
            rcpcl2_proj_new[:, :3, :4] = torch.matmul(rcpcl2_proj[:, 1, :3, :3], rcpcl2_proj[:, 0, :3])
            rcpcl1_proj_new = rcpcl1_proj[:, 0].clone()
            rcpcl1_proj_new[:, :3, :4] = torch.matmul(rcpcl1_proj[:, 1, :3, :3], rcpcl1_proj[:, 0, :3, :4])

            rcpcl1_feat = torch.cat((features[i + 1]['stage3'], lights[:, i + 1, ]), 1)
            rcpcl2_feat = features[i+2]['stage3']
            rcpcl2_feat = torch.cat((warp(rcpcl2_feat, rcpcl2_proj_new, rcpcl1_proj_new), lights[:, i+2]), 1)
            reciprocal_feats = [rcpcl1_feat.view(-1), rcpcl2_feat.view(-1)]
            pooled_reciprocal_feat, _ = torch.stack(reciprocal_feats, 1).max(1)
            pooled_reciprocal_feat = pooled_reciprocal_feat.view(shape[0], shape[1], shape[2], shape[3])

            rcpcl1_proj_new[:, :3, :4] = torch.matmul(rcpcl1_proj[:, 1, :3, :3], rcpcl1_proj[:, 0, :3])
            ref_proj_new = ref_proj[:, 0].clone()
            ref_proj_new[:, :3, :4] = torch.matmul(ref_proj[:, 1, :3, :3], ref_proj[:, 0, :3, :4])
            pooled_reciprocal_feat = warp(pooled_reciprocal_feat, rcpcl1_proj_new, ref_proj_new)

            feat, _ = self.ps_mininet(pooled_reciprocal_feat)
            feats.append(feat)

        #l_feat, shape = self.ps_feature_extractor(features[0]['stage1'])
       # r_feat, shape = self.ps_feature_extractor(features[1]['stage1'])

        pooled_feat, _ = torch.stack(feats, 1).max(1)
        #normal = self.unet(pooled_feat)
        normal = self.ps_regressor(pooled_feat, ref_feat.shape)

        return normal

    def forward(self, images, projection_mats, depth_values):
        '''

        :param self:
        :param images: shape = (B, num_src_images + 1, C, H, W)
        :param projection_mats: shape = (B, num_src_images + 1, 2, 4, 4)
        :param depth_values: [min_depth, max_depth], shape = (B, 2)
        :return:
        '''
        outputs = {}
        features = []
        lights = images[:, :, 3:, :, :]
        images = images[:, :, :3, :, :]
        for idx in range(images.shape[1]):
            img = images[:, idx]
            features.append(self.mvs_feature_extractor(img))

        if self.net_type == 'mvs' or self.net_type == 'helmholtz':
            outputs['mvs'] = self.mvs_forward(features, images[:, 0], projection_mats, depth_values)
            if self.net_type == 'mvs':
                return outputs['mvs']

        if self.net_type == 'ps' or self.net_type == 'helmholtz':
            outputs['ps'] = self.ps_forward(features, projection_mats['stage3'], lights)
            if self.net_type == 'ps':
                return outputs['ps']

        return outputs
