import os, torchvision
import numpy as np
from utils.helper_funcs import *

import torch.distributed as dist

#TODO: change to 'mps' when conv3d and other relevant functions have been implemented for mps
device_name = 'cuda' if torch.cuda.is_available() else 'cpu' if torch.backends.mps.is_available() else 'cpu'

local_rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ['WORLD_SIZE'])
is_distributed = True if world_size > 1 and device_name == 'cuda' else False

def get_summary(inputs, outputs):
    inferred_depth = outputs['stage3']['depth']
    depth_gt = inputs['depth_gts']['stage3']
    mask = inputs['masks']['stage3'].bool()

    err_map = torch.abs(depth_gt - inferred_depth) * mask.float()
    ref_images = inputs['imgs'][:, 0]

    image_summary = {"inferred_depth": inferred_depth,
                     "depth_gt": depth_gt,
                     "mask": mask,
                     "error": err_map,
                     "ref_views": ref_images
                     }

    scalar_summary = {}
    for threshold in [0.002, 0.003, 0.004, 0.02]:#[2, 3, 4, 20]:
        avg_of_avg_errs_valid, avg_accuracy = evaluate(inferred_depth, depth_gt, mask, threshold)
        scalar_summary["{}mm_avg of avg errs of valid predictions".format(int(threshold*1000))] = avg_of_avg_errs_valid
        scalar_summary["{}mm_avg accuracy".format(int(threshold*1000))] = avg_accuracy
    if is_distributed: scalar_summary = avg_summary_processes(scalar_summary)
    return dict_to_numpy(image_summary), dict_to_float(scalar_summary)

def avg_summary_processes(summary: dict):
    with torch.no_grad():
        keys = list(summary.keys())
        values = []
        for k in keys:
            values.append(summary[k])
        values = torch.stack(values, dim=0)
        dist.reduce(values, op=dist.reduce_op.SUM, dst=0)
        if local_rank == 0:
            values /= float(dist.get_world_size())
        avg_summary = {k: v for k, v in zip(keys, values)}
    return avg_summary

def add_summary(data_dict: dict, dtype: str, logger, index: int, flag: str):
    def preprocess(name, img):
        if not (len(img.shape) == 3 or len(img.shape) == 4):
            raise NotImplementedError("invalid img shape {}:{} in save_images".format(name, img.shape))
        if len(img.shape) == 3:
            img = img[:, np.newaxis, :, :]
        if img.dtype == np.bool:
            img = img.astype(np.float32)
        img = torch.from_numpy(img[:1])
        #Todo: Add value_range for the depth value
        if 'depth' in name or 'label' in name or 'gt' in name:
            return torchvision.utils.make_grid(img, padding=0, nrow=1, normalize=True, scale_each=True,
                                               value_range=(0, 2))
        elif 'mask' in name:
            return torchvision.utils.make_grid(img, padding=0, nrow=1, normalize=True, scale_each=True,
                                               value_range=(0, 1))
        elif 'error' in name:
            return torchvision.utils.make_grid(img, padding=0, nrow=1, normalize=True, scale_each=True,
                                               value_range=(0, 4))
        return torchvision.utils.make_grid(img, padding=0, nrow=1, normalize=True, scale_each=True,)

    if is_distributed and local_rank != 0:
        return

    if dtype == 'image':
        for k, v in data_dict.items():
            logger.add_image('{}/{}'.format(flag, k), preprocess(k, v), index)

    elif dtype == 'scalar':
        for k, v in data_dict.items():
            logger.add_scalar('{}/{}'.format(flag, k), v, index)

    elif dtype == 'scalars':
        logger.add_scalars('train and val', data_dict, index)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    #run:  tensorboard --logdir=/Users/culsu/Documents/UNI_stuff/Surrey/Courses/MSC_Project/src/runs
    from torch.utils.tensorboard import SummaryWriter
    recorder  = SummaryWriter('/Users/culsu/Documents/UNI_stuff/Surrey/Courses/MSC_Project/src/runs')
    import random
    randomlist = random.sample(range(10, 30), 15)
    randomlist2 = random.sample(range(10, 30), 15)