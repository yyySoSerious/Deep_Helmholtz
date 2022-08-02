import torch
from bisect import bisect_right
from torch.optim.lr_scheduler import LambdaLR
import numpy as np


def get_step_schedule_with_warmup(optimizer, milestones, gamma=0.1, warmup_factor=1.0/3, warmup_iters=500, last_epoch=-1,):
    def lr_lambda(current_step):
        if current_step < warmup_iters:
            alpha = float(current_step) / warmup_iters
            current_factor = warmup_factor * (1. - alpha) + alpha
        else:
            current_factor = 1.

        return max(0.0,  current_factor * (gamma ** bisect_right(milestones, current_step)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def save_checkpoint_func(model, optimizer, scheduler):
    def save_checkpoint(checkpoint_path, epoch):
        torch.save({'epoch': epoch + 1,
                    'model': model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()}, f'{checkpoint_path}/model_{epoch + 1:06d}.ckpt')

    return save_checkpoint


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, device=torch.device('cpu')):
    state_dict = torch.load(checkpoint_path, device)
    model.load_state_dict(state_dict['model'])
    optim_state = state_dict['optimizer']
    scheduler_state = state_dict['scheduler']
    if optimizer:
        optimizer.load_state_dict(optim_state)
    if scheduler:
        scheduler.load_state_dict(scheduler_state)

    return state_dict['epoch'], optim_state, scheduler_state


def print_dict(data: dict, prefix: str= ''):
    for k, v in data.items():
        if isinstance(v, dict):
            print_dict(v, prefix + '.' + k)
        elif isinstance(v, list):
            print(prefix+'.'+k, v)
        else:
            print(prefix+'.'+k, v.shape, f'device: {v.device}')


def dict_to_device(data: dict, device):
    new_dic = {}
    for k, v in data.items():
        if isinstance(v, dict):
            v = dict_to_device(v, device)
        elif isinstance(v, torch.Tensor):
            v = v.to(device)
        new_dic[k] = v

    return new_dic

def dict_to_numpy(data: dict):
    new_dic = {}
    for k, v in data.items():
        if isinstance(v, dict):
            v = dict_to_numpy(v)
        elif isinstance(v, list):
            v = [v_.detach().cpu().numpy().copy() for v_ in v if isinstance(v_, torch.Tensor)]
        elif isinstance(v, torch.Tensor):
            v = v.detach().cpu().numpy().copy()
        new_dic[k] = v
    return new_dic


def dict_to_float(data: dict):
    new_dic = {}
    for k, v in data.items():
        if isinstance(v, dict):
            v = dict_to_float(v)
        elif isinstance(v, torch.Tensor):
            v = v.detach().cpu().item()
        new_dic[k] = v
    return new_dic


def evaluate(batch_depth_pred, batch_depth_gt, batch_mask, threshold):
    '''
    Returns average of averages of batch accuracy and batch error in predictions that passes threshold test
    :param batch_depth_pred: (B, H, W)
    :param batch_depth_gt: (B, H, W)
    :param batch_mask: (B, H, W)
    :param threshold:
    :return:
    '''
    def evaluate_image(depth_pred, depth_gt, mask):
        error = torch.abs(depth_gt - depth_pred)
        valid = error <= threshold
        valid_avg_error = torch.mean(error[valid])
        accuracy = valid.sum(dtype=torch.float) / mask.sum(dtype=torch.float)
        return valid_avg_error, accuracy

    batch_valid_avg_error = []
    batch_accuracy = []
    for depth_pred, depth_gt, mask in zip(batch_depth_pred, batch_depth_gt, batch_mask):
        valid_avg_error, accuracy = evaluate_image(depth_pred, depth_gt, mask)
        batch_valid_avg_error.append(valid_avg_error)
        batch_accuracy.append(accuracy)

    batch_valid_avg_error = torch.stack(batch_valid_avg_error)
    batch_accuracy = torch.stack(batch_accuracy)
    return batch_valid_avg_error.mean(), batch_accuracy.mean()

class DictAverageMeter(object):
    def __init__(self):
        self.data = {}
        self.count = 0

    def update(self, new_input: dict):
        self.count += 1
        for k, v in new_input.items():
            assert isinstance(v, float), type(v)
            self.data[k] = self.data.get(k, 0) + v

    def mean(self):
        return {k: v / self.count for k, v in self.data.items()}


#todo:
def calNormalAcc(gt_n, pred_n, mask=None):
    """Tensor Dim: NxCxHxW"""
    dot_product = (gt_n * pred_n).sum(1).clamp(-1,1)
    error_map   = torch.acos(dot_product) # [-pi, pi]
    angular_map = error_map * 180.0 / np.pi
    angular_map = angular_map * mask.float()

    valid = mask.sum()
    ang_valid   = angular_map[mask]
    n_err_mean  = ang_valid.sum() / valid
    value = {'n_err_mean': n_err_mean}
    return value