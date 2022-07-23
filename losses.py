import torch
import torch.nn.functional as F


def mvs_multi_stage_loss(outputs, depth_gts, masks, weights):
    total_loss = 0
    for stage_idx in range(3):
        inferred_depth = outputs[f'stage{stage_idx+1}']['depth']
        depth_gt = depth_gts[f'stage{stage_idx+1}']
        mask = masks[f'stage{stage_idx+1}'].bool()
        depth_loss = F.smooth_l1_loss(inferred_depth[mask], depth_gt[mask], reduction='mean')
        total_loss += depth_loss * weights[stage_idx]

    return total_loss

def ps_cos_similarity_loss(output, normal_gt, mask):
    mask = mask.bool()
    #mask = mask.unsqueeze(1).repeat(1, 3, 1, 1).permute(0, 2, 3, 1)
    normal_pre = output.permute(0, 2, 3, 1)[mask]
    normal_label = normal_gt.permute(0, 2, 3 ,1)[mask]
    option = torch.ones(normal_label.shape[0], device=normal_pre.device)
    normal_loss = F.cosine_embedding_loss(normal_pre, normal_label, option)

    return normal_loss