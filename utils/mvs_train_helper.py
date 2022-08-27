import time, gc, sys, os
import torch.nn.functional as F
import torch.nn as nn
from torch.distributed.optim import ZeroRedundancyOptimizer
from ray import tune
from torch import autograd

from networks.mvs_network import MVSNet
from utils.train_helper import Helper, Config
from utils.utils import *

class MVSHelper(Helper):
    def __init__(self, args, helper_args, config, **kwargs):
        planes = (config['num_planes1'], config['num_planes2'], config['num_planes3'])
        planes_in_stages = planes[:config['num_stages']]
        conf_lambda = config['conf_lambda']
        self.num_stages = len(planes_in_stages)
        self.refine_depth = args.refine_depth
        super(MVSHelper, self).__init__(args, helper_args, config, num_stages=self.num_stages)
        self.effective_batch = config['batch'] if args.mvsalt else 4
        self.loss_weights = list(map(float, self.args.loss_weights.split(',')))
        optim_state, scheduler_state, scaler_state = None, None, None

        self.model = MVSNet(planes_in_stages=planes_in_stages,
                 conf_lambda=conf_lambda, refine_depth=self.refine_depth, conc_light=not args.no_light)

        #print(self.model)
        #fsdfd
        current_index = -1
        if self.args.ckpt_to_continue:
            current_index = int(open(self.args.ckpt_to_continue).read())
            if current_index > 0:
                model_ckpt = f'model_{current_index:06d}.ckpt'
                print(f'Loading model {model_ckpt} to continue training...')
                self.initial_epoch, optim_state, scheduler_state, scaler_state = \
                    load_checkpoint(os.path.join(args.save_dir, 'checkpoints', model_ckpt), self.model)
                print('Model loaded successfully')

        self.model = self.model.to(self.device)

        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                          lr=config['lr'], betas=(self.args.beta_1, self.args.beta_2),
                                          weight_decay=self.args.lr_decay)
        milestones = list(
            map(lambda x: int(x) * len(self.train_loader), self.args.lr_idx.split(':')[0].split(',')))
        gamma = float(self.args.lr_idx.split(':')[1])
        self.scheduler = get_step_schedule_with_warmup(optimizer=self.optimizer, milestones=milestones, gamma=gamma)

        if self.is_distributed:
            if self.args.sync_bn:
                self.model = self.apex.parallel.convert_syncbn_model(self.model)
                self.model, self.optimizer = self.apex.amp.initialize(self.model, self.optimizer,
                                                                      opt_level=self.args.opt_level)

            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.local_rank],
                                                                   output_device=self.local_rank)
            self.optimizer = ZeroRedundancyOptimizer(self.model.parameters(), optimizer_class=torch.optim.Adam,
                                                     lr=config['lr'], betas=(self.args.beta_1, self.args.beta_2),
                                                     weight_decay=self.args.lr_decay)
            self.scheduler = get_step_schedule_with_warmup(optimizer=self.optimizer, milestones=milestones, gamma=gamma)

        else:
            self.model = nn.DataParallel(self.model, device_ids=range(torch.cuda.device_count()))

        if self.args.ckpt_to_continue and current_index > 0:
            self.optimizer.load_state_dict(optim_state)
            self.scheduler.load_state_dict(scheduler_state)
            self.scaler.load_state_dict(scaler_state)


    def initialize(self, loader_data: dict, model, optimizer, scheduler):
        pass

    def train(self, logger, log_index, epoch):
        self.model.train()
        total_loss = 0.0
        for batch_idx, sample in enumerate(self.train_loader):
            log_index += batch_idx
            initialTime = time.time()
            sample = dict_to_device(sample, self.device, self.device_name)
            # print_dict(sample, prefix='After being moved to device: ')

            # print_dict(outputs)
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.use_amp):
                outputs = self.model(sample['imgs'], sample['projection_mats'], sample['depth_values'])
                loss = self.loss(outputs, sample['depth_gts'], sample['masks'])
                total_loss += float(loss)

            if self.is_distributed:
                if (batch_idx + 1) % self.effective_batch == 0 or (batch_idx + 1) == self.num_batches_train:
                    self.scaler.scale(loss).backward()
                else:
                    with self.model.no_sync():
                        self.scaler.scale(loss).backward()
            else:
                self.scaler.scale(loss).backward()
            if (batch_idx + 1) % self.effective_batch == 0 or (batch_idx + 1) == self.num_batches_train:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)

            if (log_index % self.args.log_freq == 0) :
                image_summary, scalar_summary = self.get_summary(sample, outputs)
                if self.local_rank == 0:
                    if logger:
                        self.add_summary(image_summary, 'image', logger, index=log_index, flag='train')
                        self.add_summary(scalar_summary, 'scalar', logger, index=log_index, flag='train')
                    print("Epoch {}/{}, Iter {}/{}, lr {:.6f}, train loss {:.6f}, eval 4mm ({:.6f},"
                              " {:.6f}), time = {:.2f}".format(epoch + 1, self.args.epochs, batch_idx + 1, self.num_batches_train,
                                                               self.optimizer.param_groups[0]["lr"], float(loss),
                                                               scalar_summary["4mm_mae"],
                                                               scalar_summary["4mm_acc"],
                                                               time.time() - initialTime))

                del scalar_summary, image_summary
        gc.collect()

        return total_loss / self.num_batches_train

    def validate(self, logger, log_index, epoch):
        self.model.eval()
        total_loss = 0.0
        avg_scalar_summary = DictAverageMeter()
        for batch_idx, sample in enumerate(self.val_loader):
            log_index += batch_idx
            sample = dict_to_device(sample, self.device, self.device_name)
            outputs = self.model(sample['imgs'], sample['projection_mats'], sample['depth_values'])
            # print_dict(outputs)
            loss = self.loss(outputs, sample['depth_gts'], sample['masks'])
            total_loss += float(loss)
            image_summary, scalar_summary = self.get_summary(sample, outputs)
            avg_scalar_summary.update(scalar_summary)

            if logger:
                if log_index % self.args.log_freq == 0 and self.local_rank == 0:
                    self.add_summary(image_summary, 'image', logger, index=log_index, flag='val')
                    self.add_summary(scalar_summary, 'scalar', logger, index=log_index, flag='val')

            del scalar_summary, image_summary

        if self.local_rank == 0:
            print(f'Epoch {epoch + 1}/{self.args.epochs}: {avg_scalar_summary.mean()}')
            if logger:
                self.add_summary(avg_scalar_summary.mean(), 'scalar', logger, index=epoch + 1, flag='brief')
        gc.collect()

        return total_loss / self.num_batches_val, avg_scalar_summary.mean()

    def loss(self, outputs, depth_gts: dict, masks: dict): #weighted l1 loss
        total_loss = 0
        for stage_idx in range(self.num_stages):
            inferred_depth = outputs[f'stage{stage_idx + 1}']['depth']
            depth_gt = depth_gts[f'stage{stage_idx + 1}']
            mask = masks[f'stage{stage_idx + 1}'].bool()
            depth_loss = F.smooth_l1_loss(inferred_depth[mask], depth_gt[mask], reduction='mean')
            total_loss += depth_loss * self.loss_weights[stage_idx]

            if self.num_stages == 1 and self.refine_depth:
                refined_depth = outputs['stage1']['refined_depth']
                refined_depth_loss = F.smooth_l1_loss(refined_depth[mask], depth_gt[mask], reduction='mean')
                total_loss = (depth_loss + refined_depth_loss)

        return total_loss

    def get_summary(self, inputs, outputs):
        inferred_depth = outputs[f'stage{self.num_stages}']['refined_depth'] if self.refine_depth else\
            outputs[f'stage{self.num_stages}']['depth']
        depth_gt = inputs['depth_gts'][f'stage{self.num_stages}']
        mask = inputs['masks'][f'stage{self.num_stages}'].bool()

        err_map = torch.abs(depth_gt - inferred_depth) * mask.float()
        ref_images = inputs['imgs'][:, 0, :3]
        depth_range = inputs['depth_values']
        image_summary = {"inferred_depth": [inferred_depth, depth_range],
                         "depth_gt": [depth_gt, depth_range],
                         "mask": mask,
                         "error": err_map,
                         "ref_views": ref_images
                         }

        scalar_summary = {}
        for threshold in [2, 3, 4, 20]:
            mean_abs_err, mean_accuracy = evaluate(inferred_depth, depth_gt, mask, threshold)
            scalar_summary['{}mm_mae'.format(threshold)] = mean_abs_err
            scalar_summary['{}mm_acc'.format(threshold)] = mean_accuracy
        if self.is_distributed: scalar_summary = self.avg_summary_processes(scalar_summary)
        return dict_to_numpy(image_summary), dict_to_float(scalar_summary)

class MVSConfig(Config):
    def __init__(self):
        super(MVSConfig, self).__init__()

        extra_config = {
            'batch': tune.choice([1, 2, 4, 8, 16]),
            'conf_lambda': tune.loguniform(0.5, 4.0),
            'num_stages': tune.choice([1, 2, 3]),
            'num_planes1': tune.choice([128, 64]),
            'num_planes2': tune.choice([64, 32, 16]),
            'num_planes3':tune.choice([16, 8]),
        }
        self.config.update(extra_config)