import os, time, gc
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.utils.tensorboard import SummaryWriter
from torch.profiler import ProfilerActivity

import flags
from utils.helper_funcs import *
from losses import *
from utils.summary_recorder import mvs_get_summary, ps_get_summary, add_summary
from networks.helmholtz_network import HelmholtzNet
from Dataset.helmholtz_dataset import Helmholtz_Dataset
from Dataset.preprocessing import makedir

args = flags.TrainFlags().parse()

#TODO: change to 'mps' when conv3d and other relevant functions have been implemented for mps
device_name = 'cuda' if torch.cuda.is_available() else 'cpu' if torch.backends.mps.is_available() else 'cpu'
device = torch.device(device_name)
profile_activity = [ProfilerActivity.CPU, ProfilerActivity.CUDA] if device_name == 'cuda' else [ProfilerActivity.CPU]
profile_schedule = torch.profiler.schedule(skip_first=10, wait=5, warmup=1, active=3, repeat=2)

torch.backends.cudnn.benchmark = True
local_rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ['WORLD_SIZE'])
is_distributed = True if world_size > 1 and device_name == 'cuda' else False
torch.manual_seed(args.seed)
if device_name == 'cuda':
    torch.cuda.manual_seed(args.seed)
    if args.sync_bn:
        import apex
        import apex.amp as amp


def print_peak_memory(prefix, device):
    if device == 0 and device_name == 'cuda':
        print(f'{prefix}: {torch.cuda.max_memory_allocated(device) // 1e6}MB')


def initialize(args):
    initial_epoch = 0
    loader_data = {}
    train_set = Helmholtz_Dataset(root_dir=args.root_dir, path_to_obj_dir_list=args.train_list,
                                  num_sel_views=args.num_sel_views)
    val_set = Helmholtz_Dataset(root_dir=args.root_dir, path_to_obj_dir_list=args.val_list,
                                  num_sel_views=args.num_sel_views)
    if is_distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')
        dist.barrier()
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, num_replicas=dist.get_world_size(),
                                                                        rank=dist.get_rank())
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_set, num_replicas=dist.get_world_size(),
                                                                        rank=dist.get_rank())
    else:
        train_sampler, val_sampler = None, None

    loader_data['train_sampler'] = train_sampler
    loader_data['val_sampler'] = val_sampler
    loader_data['train_loader'] = torch.utils.data.DataLoader(train_set, args.batch, sampler=train_sampler, num_workers=1,
                                               drop_last=True, shuffle=train_sampler is None, pin_memory=True)
    loader_data['val_loader'] = torch.utils.data.DataLoader(val_set, args.batch, sampler=val_sampler, num_workers=1,
                                               drop_last=True, shuffle=val_sampler is None,
                                                            pin_memory=device_name != 'cpu')

    model = HelmholtzNet(planes_in_stages=list(map(int, args.planes_in_stages.split(','))),
                         conf_lambda=args.conf_lambda, net_type=args.net_type)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=args.lr, betas=(args.beta_1, args.beta_2), weight_decay=args.lr_decay)
    milestones = list(map(lambda x: int(x) * len(loader_data['train_loader']), args.lr_idx.split(':')[0].split(',')))
    gamma = float(args.lr_idx.split(':')[1])
    scheduler = get_step_schedule_with_warmup(optimizer=optimizer, milestones=milestones, gamma=gamma)

    if args.ckpt_to_continue:
        print(f'Loading model {args.ckpt_to_continue} to continue training...')
        initial_epoch = load_checkpoint(args.ckpt_to_continue, model, optimizer, scheduler)
        print('Model loaded successfully')

    model = model.to(device)
    if is_distributed:
        if args.sync_bn:
                model = apex.parallel.convert_syncbn_model(model)
                model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level)

        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
        optimizer = ZeroRedundancyOptimizer(model.parameters(), optimizer_class=torch.optim.Adam, lr=0.01)

    else:
        model = nn.DataParallel(model)

    print_peak_memory("Max memory allocated after creating local model", local_rank)

    return model, optimizer, scheduler, initial_epoch, loader_data


@record
def main(args, model, optimizer, scheduler, initial_epoch, loader_data):
    train_loader = loader_data['train_loader']
    train_sampler = loader_data['train_sampler']
    val_loader = loader_data['val_loader']
    val_sampler = loader_data['val_sampler']
    logger = None
    checkpoint_dir = os.path.join(args.save_dir, 'checkpoints')
    log_dir = os.path.join(args.save_dir, 'logs')
    if local_rank == 0:
        makedir(checkpoint_dir)
        makedir(log_dir)
        logger = SummaryWriter(log_dir)
        print(args)

    loss_weights = list(map(float, args.loss_weights.split(',')))
    num_batches_train = len(train_loader)
    num_batches_val = len(val_loader)

    train = train_func(args, train_loader, model, optimizer, scheduler, loss_weights)
    validate =  validate_func(args, val_loader, model, loss_weights)
    save_checkpoint = save_checkpoint_func(model, optimizer, scheduler)

    losses = {}

    with torch.profiler.profile(activities=profile_activity, schedule=profile_schedule, profile_memory=True,
                                record_shapes=True,
                                on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir)) as prof:
        for epoch in np.arange(initial_epoch, args.epochs):
            if is_distributed:
                train_sampler.set_epoch(epoch)
                val_sampler.set_epoch(epoch)

            #train
            log_index = (num_batches_train + num_batches_val) * epoch
            losses['train_loss'] = train(logger, log_index, epoch, prof)

            #save snapshot of model
            if local_rank == 0 and (epoch + 1) * args.save_freq == 0:
                save_checkpoint(checkpoint_dir, epoch)

            #validation
            if(epoch + 1) % args.eval_freq == 0 or (epoch +1) == args.epochs:
                with torch.no_grad():
                    log_index = num_batches_train * (epoch + 1) + num_batches_val * epoch
                    losses['val_loss'] = validate(logger, log_index, epoch)

            #record train and validation loss
            if local_rank == 0:
                logger.add_scalars('train and val losses', losses, epoch+1)

    dist.destroy_process_group()

#TODO : test other images
def train_func(args, train_loader, model, optimizer, scheduler, loss_weights):
    num_batches_train = len(train_loader)
    def train(logger, log_index, epoch, profiler):
        model.train()
        total_loss = 0.0
        loss = None
        for batch_idx, sample in enumerate(train_loader):
            log_index += batch_idx
            initialTime = time.time()
            sample = dict_to_device(sample, device)
            # print_dict(sample, prefix='After being moved to device: ')

            optimizer.zero_grad(set_to_none=True)

            outputs = model(sample['imgs'], sample['projection_mats'], sample['depth_values'])
            profiler.step()
            # print_dict(outputs)
            if(args.net_type == 'mvs'):
                loss = mvs_multi_stage_loss(outputs, sample['depth_gts'], sample['masks'], loss_weights)
            elif (args.net_type == 'ps'):
                loss = ps_cos_similarity_loss(outputs, sample['normal_gt'], sample['masks']['stage3'])
            total_loss += loss.item()
            if is_distributed and args.sync_bn:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            optimizer.step()
            scheduler.step()

            if log_index % args.log_freq == 0:
                image_summary, scalar_summary = None, None
                if args.net_type == 'mvs':
                    image_summary, scalar_summary = mvs_get_summary(sample, outputs)
                elif args.net_type == 'ps':
                    image_summary, scalar_summary = ps_get_summary(sample, outputs)
                if local_rank == 0:
                    add_summary(image_summary, 'image', logger, index=log_index, flag='train')
                    add_summary(scalar_summary, 'scalar', logger, index=log_index, flag='train')
                    if args.net_type == 'mvs' or args.net_type == 'helmholtz':
                        print("Epoch {}/{}, Iter {}/{}, lr {:.6f}, train loss {:.6f}, eval 4mm ({:.6f},"
                              " {:.6f}), time = {:.2f}".format(epoch + 1, args.epochs, batch_idx + 1, num_batches_train,
                                                               optimizer.param_groups[0]["lr"], loss,
                                                               scalar_summary["4mm_avg of avg errs of valid predictions"],
                                                               scalar_summary["4mm_avg accuracy"],
                                                               time.time() - initialTime))
                    elif args.net_type == 'ps' or args.net_type == 'helmholtz':
                        print("Epoch {}/{}, Iter {}/{}, lr {:.6f}, train loss {:.6f}, n_err_mean ({:.6f}),"
                              " time = {:.2f}".format(epoch + 1, args.epochs, batch_idx + 1, num_batches_train,
                                                               optimizer.param_groups[0]["lr"], loss,
                                                               scalar_summary['n_err_mean'],
                                                               time.time() - initialTime))


                del scalar_summary, image_summary

        gc.collect()

        return total_loss / num_batches_train

    return train


def validate_func(args, val_loader, model, loss_weights):
    num_batches_val = len(val_loader)
    def validate(logger, log_index, epoch):
        model.eval()
        total_loss = 0.0
        avg_scalar_summary = DictAverageMeter()
        for batch_idx, sample in enumerate(val_loader):
            log_index += batch_idx
            sample = dict_to_device(sample, device)
            outputs = model(sample['imgs'], sample['projection_mats'], sample['depth_values'])
            loss = mvs_multi_stage_loss(outputs, sample['depth_gts'], sample['masks'], loss_weights)
            total_loss += loss.item()

            if args.net_type == 'mvs':
                image_summary, scalar_summary = mvs_get_summary(sample, outputs)
            elif args.net_type == 'ps':
                image_summary, scalar_summary = ps_get_summary(sample, outputs)
            avg_scalar_summary.update(scalar_summary)

            if log_index % args.log_freq == 0 and local_rank == 0:
                add_summary(image_summary, 'image', logger, index=log_index, flag='val')
                add_summary(scalar_summary, 'scalar', logger, index=log_index, flag='val')

            del scalar_summary, image_summary

        if local_rank == 0:
            print(f'Epoch {epoch +1}/{args.epochs}: {avg_scalar_summary.mean()}')
            add_summary(avg_scalar_summary.mean(), 'scalar', logger, index=epoch + 1, flag='brief')

        gc.collect()

        return total_loss / num_batches_val

    return validate




if __name__=='__main__':
    model, optimizer, scheduler, epoch, loader_data = initialize(args)
    main(args, model, optimizer, scheduler, epoch, loader_data)