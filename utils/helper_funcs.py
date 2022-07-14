import torch
from bisect import bisect_right
from torch.optim.lr_scheduler import LambdaLR


def get_step_schedule_with_warmup(optimizer, milestones, gamma=0.1, warmup_factor=1.0/3, warmup_iters=500, last_epoch=-1,):
    def lr_lambda(current_step):
        if current_step < warmup_iters:
            alpha = float(current_step) / warmup_iters
            current_factor = warmup_factor * (1. - alpha) + alpha
        else:
            current_factor = 1.

        return max(0.0,  current_factor * (gamma ** bisect_right(milestones, current_step)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


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

