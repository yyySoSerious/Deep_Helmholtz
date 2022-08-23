import torch, gc
import torch.nn.functional as F
from utils.grid_sample_gradfix import grid_sample

eps = 1e-12


def uncertainty_aware_samples(curr_depth, expected_variance, num_depth, dtype, device, shape):
    # first stage
    if curr_depth.dim() == 2:
        curr_depth_min = curr_depth[:, 0]  # shape: (B, )
        curr_depth_max = curr_depth[:, 1]  # shape: (B, )
        new_interval = (curr_depth_max - curr_depth_min) / (num_depth - 1)  # shape: (B, )
        depth_range_samples = curr_depth_min.unsqueeze(1) + (torch.arange(0, num_depth, device=device, dtype=dtype,
                                                                          requires_grad=False).reshape(1, -1) *
                                                             new_interval.unsqueeze(1))  # shape: (B, D)
        depth_range_samples = depth_range_samples.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, shape[1],
                                                                                     shape[2])  # shape (B, D, H, W)

    # Not the first stage
    else:
        left_shift = -torch.min(curr_depth, expected_variance)
        right_shift = expected_variance

        assert num_depth > 1

        step = (right_shift - left_shift) / (float(num_depth) - 1)
        new_depth_samples = []
        for i in range(int(num_depth)):
            new_depth_samples.append(curr_depth + left_shift + step * i + eps)

        depth_range_samples = torch.cat(new_depth_samples, 1)

    return depth_range_samples


# noinspection PyTypeChecker
def compute_depth(curr_stage_features, curr_stage_projection_mats, depth_samples, cost_reg, conf_lambda,
                  is_training=False):
    '''

    :param curr_stage_features:
    :param curr_stage_projection_mats:
    :param depth_samples:
    :param cost_reg:
    :param conf_lambda:
    :param is_training:
    :return:
    '''

    curr_stage_projection_mats = torch.unbind(curr_stage_projection_mats, 1)
    num_views = len(curr_stage_features)
    num_depths = depth_samples.shape[1]

    assert len(
        curr_stage_projection_mats) == num_views, \
        'Number of images and number of projection matrices ds are not the same'

    ref_feature, src_features = curr_stage_features[0], curr_stage_features[1:]
    ref_projection_mat, src_projection_mats = curr_stage_projection_mats[0], curr_stage_projection_mats[1:]

    ref_volume = ref_feature.unsqueeze(2).repeat(1, 1, num_depths, 1, 1)
    volume_sum = ref_volume
    volume_sq_sum = ref_volume ** 2
    del ref_volume

    for src_feat, src_proj in zip(src_features, src_projection_mats):
        src_proj_new = src_proj[:, 0].clone()
        src_proj_new[:, :3, :4] = torch.matmul(src_proj[:, 1, :3, :3], src_proj[:, 0, :3])

        ref_proj_new = ref_projection_mat[:, 0].clone()
        ref_proj_new[:, :3, :4] = torch.matmul(ref_projection_mat[:, 1, :3, :3], ref_projection_mat[:, 0, :3, :4])
        warped_volume = homo_warping(src_feat, src_proj_new, ref_proj_new, depth_samples)

        if is_training:
            volume_sum = volume_sum + warped_volume
            volume_sq_sum = volume_sq_sum + warped_volume ** 2
        else:
            volume_sum += warped_volume
            volume_sq_sum += warped_volume.pow_(2)  # in_place method
        del warped_volume
    gc.collect()
    volume_variance = volume_sq_sum.div_(num_views).sub_(volume_sum.div_(num_views).pow_(2))

    prob_volume_pre = cost_reg(volume_variance).squeeze(1)
    prob_volume = F.softmax(prob_volume_pre, dim=1)
    depth = depth_regression(prob_volume, depth_samples=depth_samples)

    with torch.no_grad():
        prob_volume_sum4 = 4 * F.avg_pool3d(F.pad(prob_volume.unsqueeze(1), pad=(0, 0, 0, 0, 1, 2)), (4, 1, 1),
                                            stride=1, padding=0).squeeze(1)
        depth_index = depth_regression(prob_volume, depth_samples=torch.arange(num_depths, device=prob_volume.device,
                                                                               dtype=torch.float)).long()
        depth_index = depth_index.clamp(min=0, max=num_depths - 1)
        prob_conf = torch.gather(prob_volume_sum4, 1, depth_index.unsqueeze(1)).squeeze(1)

    variance_samples = (depth_samples - depth.unsqueeze(1)) ** 2
    expected_variance = conf_lambda * torch.sum(variance_samples * prob_volume, dim=1, keepdim=False) ** 0.5

    return {'depth': depth, 'confidence': prob_conf, 'variance': expected_variance}


def homo_warping(src_feat, src_proj, ref_proj, depth_samples):
    '''
    :param src_feat:  (B, C, H, W)
    :param src_proj: (B, 4, 4)
    :param ref_proj: (B, 4, 4)
    :param depth_samples: (B, num_depths, H, W)
    :return: (B, C, num_depth, H, W)
    '''
    batch, channels = src_feat.shape[0], src_feat.shape[1]
    num_depths = depth_samples.shape[1]
    height, width = src_feat.shape[2], src_feat.shape[3]

    with torch.no_grad():
        proj = torch.matmul(src_proj, torch.inverse(ref_proj))
        rot = proj[:, :3, :3]  # shape: (B, 3, 3)
        trans = proj[:, :3, 3:4]  # shape: (B, 3, 1)

        y, x = torch.meshgrid(torch.arange(0, height, dtype=torch.float32, device=src_feat.device),
                              torch.arange(0, width, dtype=torch.float32, device=src_feat.device), indexing='ij')
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(height * width), x.view(height * width)  # shape: (HxW,), (HxW,)
        xyz = torch.stack((x, y, torch.ones_like(x)))  # shape: (3, H*W)
        xyz = xyz.unsqueeze(0).repeat(batch, 1, 1)  # shape: (B, 3, H*W)
        rot_xyz = rot.matmul(xyz)  # shape: (B, 3, H*W]
        rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, num_depths, 1) * \
                        depth_samples.view(batch, 1, num_depths, -1)  # shape: (B, 3, num_depths, H*W)
        proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1, 1)  # shape: (B, 3, num_depths, H*W)
        proj_xy = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :]  # shape: (B, 2, num_depths, H*W)
        proj_x_normalized = proj_xy[:, 0, :, :] / ((width - 1) / 2) - 1  # shape: (B, num_depths, H*W)
        proj_y_normalized = proj_xy[:, 1, :, :] / ((height - 1) / 2) - 1  # shape: (B, num_depths, H*W)
        proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # shape: (B, num_depths, H*W, 2)
        grid = proj_xy
        del proj_x_normalized, proj_y_normalized


    warped_src_feat = grid_sample(src_feat, grid.view(batch, num_depths * height, width, 2)) #F.grid_sample(src_feat, grid.view(batch, num_depths * height, width, 2), mode='bilinear',
                                    #padding_mode='zeros', align_corners=False)

    warped_src_feat = warped_src_feat.view(batch, channels, num_depths, height, width)

    return warped_src_feat


def depth_regression(prob_volume, depth_samples):
    if depth_samples.dim() <= 2:
        depth_samples = depth_samples.view(*depth_samples.shape, 1, 1)
    depth = torch.sum(prob_volume * depth_samples, 1)

    return depth

def refine_depth(init_depth_map, ref_img, depth_values, shape, depth_refiner):
    min_depth = depth_values[:, 0]
    max_depth = depth_values[:, 1]
    min_depth_2d = min_depth.unsqueeze(-1).unsqueeze(-1).repeat(1, shape[0], shape[1])
    max_depth_2d = max_depth.unsqueeze(-1).unsqueeze(-1).repeat(1, shape[0], shape[1])
    depth_scale_2d = max_depth_2d - min_depth_2d
    init_norm_depth_map = (init_depth_map - min_depth_2d).div_(depth_scale_2d)
    norm_depth_map = depth_refiner(ref_img, init_norm_depth_map)
    refined_depth_map = norm_depth_map.mul_(depth_scale_2d) + min_depth_2d

    return refined_depth_map


def warp(src_feat, src_proj, ref_proj):
    batch, channels = src_feat.shape[0], src_feat.shape[1]
    height, width = src_feat.shape[2], src_feat.shape[3]

    with torch.no_grad():
        proj = torch.matmul(src_proj, torch.inverse(ref_proj))
        rot = proj[:, :3, :3]  # shape: (B, 3, 3)
        trans = proj[:, :3, 3:4]  # shape: (B, 3, 1)

        y, x = torch.meshgrid(torch.arange(0, height, dtype=torch.float32, device=src_feat.device),
                              torch.arange(0, width, dtype=torch.float32, device=src_feat.device), indexing='ij')
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(height * width), x.view(height * width)  # shape: (HxW,), (HxW,)
        xyz = torch.stack((x, y, torch.ones_like(x)))  # shape: (3, H*W)
        xyz = xyz.unsqueeze(0).repeat(batch, 1, 1)  # shape: (B, 3, H*W)
        rot_xyz = rot.matmul(xyz)  # shape: (B, 3, H*W]
        proj_xyz = rot_xyz + trans  # shape: (B, 3, H*W)
        proj_xy = proj_xyz[:, :2, :] / proj_xyz[:, 2:3, :]  # shape: (B, 2, H*W)
        proj_x_normalized = proj_xy[:, 0, :] / ((width - 1) / 2) - 1  # shape: (B, H*W)
        proj_y_normalized = proj_xy[:, 1, :] / ((height - 1) / 2) - 1  # shape: (B, H*W)
        proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=2)  # shape: (B, H*W, 2)
        grid = proj_xy

    warped_src_feat = grid_sample(src_feat, grid.view(batch, height, width, 2)) #F.grid_sample(src_feat, grid.view(batch, height, width, 2), mode='bilinear',
                                    #padding_mode='zeros', align_corners=False)

    return warped_src_feat.view(batch, channels, height, width)