import time, gc
import torch.nn.functional as F
import torch.nn as nn
from torch.distributed.optim import ZeroRedundancyOptimizer

from networks.ps_network import PsNet
from utils.helper import Helper
from utils.utils import *

class PSHelper(Helper):
    def __init__(self, args, helper_args, config, in_channels=6):
        super(PSHelper, self).__init__(args, helper_args, config)
        optim_state, scheduler_state = None, None
        self.model = PsNet(in_channels, self.args.base_channels, fuse_type=config['fuse_type'],
                           rcpcl_fuse_type=config['rcpcl_fuse_type'], add_type=config['add_type'], bn=config['use_bn'],
                           nn_layers=config['nn_layers'])

        #print(self.model)

        if self.args.ckpt_to_continue:
            print(f'Loading model {self.args.ckpt_to_continue} to continue training...')
            self.initial_epoch, optim_state, scheduler_state = load_checkpoint(self.args.ckpt_to_continue, self.model)
            print('Model loaded successfully')

        self.model = self.model.to(self.device)

        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                     lr=config['lr'], betas=(self.args.beta_1, self.args.beta_2), weight_decay=self.args.lr_decay)
        milestones = list(
            map(lambda x: int(x) * len(self.train_loader), self.args.lr_idx.split(':')[0].split(',')))
        gamma = float(self.args.lr_idx.split(':')[1])
        self.scheduler = get_step_schedule_with_warmup(optimizer=self.optimizer, milestones=milestones, gamma=gamma)

        if self.is_distributed:
            if self.args.sync_bn:
                self.model = self.apex.parallel.convert_syncbn_model(self.model)
                self.model, self.optimizer = self.apex.amp.initialize(self.model, self.optimizer, opt_level=self.args.opt_level)

            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.local_rank],
                                                                   output_device=self.local_rank)
            self.optimizer = ZeroRedundancyOptimizer(self.model.parameters(), optimizer_class=torch.optim.Adam,
                                                     lr=config['lr'], betas=(self.args.beta_1, self.args.beta_2), weight_decay=self.args.lr_decay)
            self.scheduler = get_step_schedule_with_warmup(optimizer=self.optimizer, milestones=milestones, gamma=gamma)

        else:
            self.model = nn.DataParallel(self.model)

        if self.args.ckpt_to_continue:
            self.optimizer.load_state_dict(optim_state)
            self.scheduler.load_state_dict(scheduler_state)

    def initialize(self, loader_data: dict, model, optimizer, scheduler):
        pass

    def train(self, logger, log_index, epoch):
        self.model.train()
        total_loss = 0.0
        for batch_idx, sample in enumerate(self.train_loader):
            log_index += batch_idx
            initialTime = time.time()
            sample = dict_to_device(sample, self.device)
            # print_dict(sample, prefix='After being moved to device: ')

            self.optimizer.zero_grad(set_to_none=True)
            outputs = self.model(sample['imgs'], sample['projection_mats'])

            # print_dict(outputs)
            loss = self.loss(outputs, sample['normal_gt'], sample['mask'])
            total_loss += loss.item()
            if self.is_distributed and self.args.sync_bn:
                with self.apex.amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            self.optimizer.step()
            self.scheduler.step()

            if (log_index % self.args.log_freq == 0) and self.local_rank == 0:
                image_summary, scalar_summary = self.get_summary(sample, outputs)
                print("Epoch {}/{}, Iter {}/{}, lr {:.6f}, train loss {:.6f}, n_err_mean ({:.6f}),"
                          " time = {:.2f}".format(epoch + 1, self.args.epochs, batch_idx + 1, self.num_batches_train,
                                                           self.optimizer.param_groups[0]["lr"], loss,
                                                           scalar_summary['n_err_mean'],
                                                           time.time() - initialTime))
                if logger:
                    self.add_summary(image_summary, 'image', logger, index=log_index, flag='train')
                    self.add_summary(scalar_summary, 'scalar', logger, index=log_index, flag='train')
                del scalar_summary, image_summary
        gc.collect()

        return total_loss / self.num_batches_train

    def validate(self, logger, log_index, epoch):
        self.model.eval()
        total_loss = 0.0
        avg_scalar_summary = DictAverageMeter()
        for batch_idx, sample in enumerate(self.val_loader):
            log_index += batch_idx
            sample = dict_to_device(sample, self.device)
            outputs = self.model(sample['imgs'], sample['projection_mats'])
            # print_dict(outputs)
            loss = self.loss(outputs, sample['normal_gt'], sample['mask'])
            total_loss += loss.item()

            if logger:
                image_summary, scalar_summary = self.get_summary(sample, outputs)
                avg_scalar_summary.update(scalar_summary)

                if log_index % self.args.log_freq == 0 and self.local_rank == 0:
                    self.add_summary(image_summary, 'image', logger, index=log_index, flag='val')
                    self.add_summary(scalar_summary, 'scalar', logger, index=log_index, flag='val')

                del scalar_summary, image_summary

        if self.local_rank == 0:
            print(f"Epoch {epoch +1}/{self.args.epochs}: {avg_scalar_summary.mean()}")
            if logger:
                self.add_summary(avg_scalar_summary.mean(), 'scalar', logger, index=epoch + 1, flag='brief')
        gc.collect()

        return total_loss / self.num_batches_val, avg_scalar_summary.mean()

    def loss(self, output: torch.tensor, normal_gt: torch.tensor, mask: torch.tensor): #cos similarity loss
        mask = mask.bool()
        normal_pre = output.permute(0, 2, 3, 1)[mask]
        normal_label = normal_gt.permute(0, 2, 3, 1)[mask]
        option = torch.ones(normal_label.shape[0], device=normal_pre.device)
        normal_loss = F.cosine_embedding_loss(normal_pre, normal_label, option)

        return normal_loss

    def get_summary(self, inputs, output):
        inferred_normal = output
        normal_gt = inputs['normal_gt']
        mask = inputs['mask'].bool()

        ref_images = inputs['imgs'][:, 0, :3]

        image_summary = {"inferred_normal": inferred_normal,
                         "normal_gt": normal_gt,
                         "mask": mask,
                         "ref_views": ref_images
                         }

        scalar_summary = calc_normal_acc(normal_gt, inferred_normal, mask)
        if self.is_distributed: scalar_summary = self.avg_summary_processes(scalar_summary)

        return dict_to_numpy(image_summary), dict_to_float(scalar_summary)