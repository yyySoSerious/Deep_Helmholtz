import torch, torchvision
import numpy as np
import torch.distributed as dist
from ray import tune
import sys

from Dataset.helmholtz_dataset import Helmholtz_Dataset


class Helper:
    def __init__(self, args, helper_args, config, **kwargs):
        self.args = args
        self.max_norm = config['max_norm']
        self.local_rank = helper_args['local_rank']
        self.is_distributed = helper_args['is_distributed']
        self.initial_epoch = 1
        if self.is_distributed and self.args.sync_bn:
            import apex
            self.apex = apex

        train_set = Helmholtz_Dataset(root_dir=self.args.root_dir, path_to_obj_dir_list=self.args.train_list,
                                      num_reciprocals=self.args.num_reciprocals, net=self.args.net_type, **kwargs)
        val_set = Helmholtz_Dataset(root_dir=self.args.root_dir, path_to_obj_dir_list=self.args.val_list,
                                    num_reciprocals=self.args.num_reciprocals, net=self.args.net_type, **kwargs)

        if self.is_distributed:
            torch.cuda.set_device(self.local_rank)
            dist.init_process_group(backend='nccl')
            dist.barrier()
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(train_set,
                                                                            num_replicas=dist.get_world_size(),
                                                                            rank=dist.get_rank())
            self.val_sampler = torch.utils.data.distributed.DistributedSampler(val_set, num_replicas=dist.get_world_size(),
                                                                          rank=dist.get_rank())
        else:
            self.train_sampler, self.val_sampler = None, None

        batch = 2 if args.mvsalt else config['batch']

        self.device_name = helper_args['device_name']
        self.device = torch.device(self.device_name)
        self.use_amp = True if self.device_name == 'cuda' else False
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.train_loader = torch.utils.data.DataLoader(train_set, batch, sampler=self.train_sampler,
                                                        num_workers=args.num_gpus*2, drop_last=True,
                                                        shuffle=self.train_sampler is None,
                                                        pin_memory=self.device_name != 'cpu')
        self.val_loader = torch.utils.data.DataLoader(val_set, batch, sampler=self.val_sampler,
                                                      num_workers=args.num_gpus*2, drop_last=True,
                                                      shuffle=self.val_sampler is None,
                                                      pin_memory=self.device_name != 'cpu')

        self.num_batches_train = len(self.train_loader)
        self.num_batches_val = len(self.val_loader)

    def initialize(self, loader_data: dict, model, optimizer, scheduler):
        pass

    def train(self, logger, log_index, epoch):
        pass

    def validate(self, logger, log_index, epoch):
        pass

    def loss(self, output, normal_gt, mask):
        pass

    def get_summary(self, inputs, output):
        pass

    def avg_summary_processes(self, summary: dict):
        with torch.no_grad():
            keys = list(summary.keys())
            values = []
            for k in keys:
                values.append(summary[k])
            values = torch.stack(values, dim=0)
            dist.reduce(values, op=dist.ReduceOp.SUM, dst=0)
            if self.local_rank == 0:
                values /= float(dist.get_world_size())
            avg_summary = {k: v for k, v in zip(keys, values)}
        return avg_summary

    def add_summary(self, data_dict: dict, dtype: str, logger, index: int, flag: str):
        def preprocess(name, img):
            depth_range = None
            if 'depth' in name or 'label' in name:
                depth_range = img[1][0]
                img = img[0]
            if not (len(img.shape) == 3 or len(img.shape) == 4):
                raise NotImplementedError("invalid img shape {}:{} in save_images".format(name, img.shape))
            if len(img.shape) == 3:
                img = img[:, np.newaxis, :, :]
            if img.dtype == np.bool:
                img = img.astype(np.float32)
            img = torch.from_numpy(img[:1])


            if 'depth' in name or 'label' in name:
                return torchvision.utils.make_grid(img, padding=0, nrow=1, normalize=True, scale_each=True,
                                                   value_range=tuple(depth_range))
            elif 'mask' in name:
                return torchvision.utils.make_grid(img, padding=0, nrow=1, normalize=True, scale_each=True,
                                                   value_range=(0, 1))
            elif 'error' in name:
                return torchvision.utils.make_grid(img, padding=0, nrow=1, normalize=True, scale_each=True,
                                                   value_range=(0, 4))
            elif 'normal' in name:
                return torchvision.utils.make_grid(img, padding=0, nrow=1, normalize=True, scale_each=True,
                                                   value_range=(-1, 1))
            return torchvision.utils.make_grid(img, padding=0, nrow=1, normalize=True, scale_each=True, )

        if self.is_distributed and self.local_rank != 0:
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

    def getBack(var_grad_fn):
        print(var_grad_fn, file=sys.stderr)
        for n in var_grad_fn.next_functions:
            if n[0]:
                try:
                    tensor = getattr(n[0], 'variable')
                    print(n[0], file=sys.stderr)
                    print('Tensor with grad found:', tensor, file=sys.stderr)
                    print(' - gradient:', tensor.grad, file=sys.stderr)
                    print(file=sys.stderr)
                except AttributeError as e:
                    Helper.getBack(n[0])

class Config:
    def __init__(self):
        self.config = {
            'batch': tune.choice([1, 2, 4, 8]),
            'lr': tune.loguniform(1e-4, 1e-3), #5e-3, 5e-2)
            'max_norm': tune.uniform(1, 5),
        }