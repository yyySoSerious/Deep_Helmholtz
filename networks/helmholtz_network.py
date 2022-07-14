import torch.nn as nn
from . import mvs_network as mvs
from . import ps_network as ps
from .net_functions import *


class HelmholtzNet(nn.Module):
    def __init__(self, planes_in_stages=(64, 32, 8), conf_lambda=1.5, grad_method='detach',
                 base_channels_per_stage=(8, 8, 8)):
        super(HelmholtzNet, self).__init__()

        self.planes_in_stages = planes_in_stages
        self.conf_lambda = conf_lambda
        self.grad_method = grad_method
        self.base_channels_per_stage = base_channels_per_stage
        self.num_stages = len(planes_in_stages)
        self.scale_factors_inv = {'stage1': 4.0, 'stage2': 2.0, 'stage3': 1.0}

        self.mvs_feature_extractor = mvs.FeatExtractorNet(base_channels=base_channels_per_stage[0],
                                                          num_stages=self.num_stages)
        self.mvs_cost_regularizer = nn.ModuleList(
            [mvs.CostRegularizerNet(in_channels=self.mvs_feature_extractor.out_channels[i],
                                    base_channels=self.base_channels_per_stage[i]) for i in range(self.num_stages)])
        self.ps_feature_extractor = ps.FeatExtractorNet(base_channels=base_channels_per_stage[0])
        self.ps_regressor = ps.RegressionNet(base_channels=base_channels_per_stage[0])

    def forward(self, images, projection_mats, depth_values):
        '''

        :param self:
        :param images: shape = (B, num_src_images + 1, C, H, W)
        :param projection_mats: shape = (B, num_src_images + 1, 2, 4, 4)
        :param depth_values: [min_depth, max_depth], shape = (B, 2)
        :return:
        '''
        features = []
        for idx in range(images.shape[1]):
            img = images[:, idx]
            features.append(self.mvs_feature_extractor(img))

        outputs = {}
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
                                               cost_reg=self.mvs_cost_regularizer[stage_num], conf_lambda=self.conf_lambda,
                                               is_training=self.training)

            depth = curr_stage_outputs['depth']
            expected_variance = curr_stage_outputs['variance']

            outputs[f'stage{stage_num + 1}'] = curr_stage_outputs

        return outputs
