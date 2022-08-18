import os, gc, time
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
from torch.distributed.elastic.multiprocessing.errors import record
from torch.utils.tensorboard import SummaryWriter
from ray import tune
from ray.tune.suggest.bohb import TuneBOHB
from ray.tune.schedulers import HyperBandForBOHB
from ray.tune import CLIReporter
from functools import partial

import flags
from utils.ps_train_helper import PSHelper, PSConfig
from utils.mvs_train_helper import MVSHelper, MVSConfig
from utils.utils import *
from Dataset.preprocessing import makedir

args = flags.TrainFlags().parse()

#TODO: change to 'mps' when conv3d and other relevant functions have been implemented for mps
device_name = 'cuda' if torch.cuda.is_available() else 'cpu' if torch.backends.mps.is_available() else 'cpu'


torch.backends.cudnn.benchmark = True
local_rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ['WORLD_SIZE'])
is_distributed = True if world_size > 1 and device_name == 'cuda' and not args.ray_tune else False
torch.manual_seed(args.seed)
if device_name == 'cuda':
    torch.cuda.manual_seed(args.seed)
    if args.sync_bn:
        import apex.amp as amp


def print_peak_memory(prefix, device):
    if device == 0 and device_name == 'cuda':
        print(f'{prefix}: {torch.cuda.max_memory_allocated(device) // 1e6}MB')

@record
def main(args):
    logger = None
    checkpoint_dir = os.path.join(args.save_dir, 'checkpoints')
    log_dir = os.path.join(args.save_dir, 'logs')
    if local_rank == 0:
        makedir(checkpoint_dir)
        makedir(log_dir)
        logger = SummaryWriter(log_dir)
        print(args)

    if args.ray_tune:
        def trial_name_string(trial):
            """
            Args:
                trial (Trial): A generated trial object.

            Returns:
                trial_name (str): String representation of Trial.
            """
            return str(trial)

        helper_args = {'device_name': device_name, 'local_rank': local_rank, 'is_distributed': is_distributed}
        config = None
        if args.net_type == 'ps':
            config = PSConfig().config
        elif args.net_type == 'mvs':
            config = MVSConfig().config

        algo = TuneBOHB(metric="loss", mode="min")
        bohb = HyperBandForBOHB(
            time_attr="training_iteration",
            metric="loss",
            mode="min",
            max_t=args.epochs)

        reporter = CLIReporter(
            metric_columns=['loss', 'info', 'training_iteration'])
        analysis = tune.run(
            partial(train_tune, args=args, helper_args=helper_args),
            local_dir=args.save_dir,
            name=args.experiment_name,
            num_samples=10,
            resources_per_trial= {'gpu': args.num_gpus, 'cpu': args.num_workers*2} if device_name == 'cuda' else {'cpu': 2, 'memory': 2 * 1024 * 1024}, #,
            resume='AUTO',
            config=config, scheduler=bohb, search_alg=algo,
            progress_reporter=reporter,)

        best_trial = analysis.get_best_trial('loss', 'min', 'last')
        best_trial_path = os.path.join(args.save_dir, args.experiment_name, 'best_trial')
        makedir(best_trial_path)
        with open(best_trial_path, 'w') as f:
            f.write('Best trial config: {}'.format(best_trial.config))
            f.write('Best trial final validation  loss: {}'.format(best_trial.last_result['loss']))
            f.write('Best trial final accuracy info: {} '.format(best_trial.last_resutlt['info']))
    else:
        train(args,logger, checkpoint_dir)


    torch.distributed.destroy_process_group()

def train(args, logger, checkpoint_dir):
    print('Initializing...')
    helper = None
    helper_args = {'device_name': device_name, 'local_rank': local_rank, 'is_distributed': is_distributed}
    planes_in_stages = list(map(int, args.planes_in_stages.split(',')))
    num_stages = len(planes_in_stages)
    nn_layers = list(map(int, args.nn_layers.split(',')))
    config = {'base_channels': args.base_channels, 'batch': args.batch, 'lr': args.lr, 'use_bn': args.use_bn,
              'fuse_type': args.fuse_type, 'rcpcl_fuse_type': args.rcpcl_fuse_type, 'add_type': args.add_type,
              'nn_layers': nn_layers, 'conf_lambda': args.conf_lambda, 'num_stages':num_stages,
              'num_planes1': planes_in_stages[0], 'num_planes2': planes_in_stages[1] if num_stages > 1 else 0 ,
              'num_planes3': planes_in_stages[2] if num_stages > 2 else 0, 'max_norm': args.max_norm}

    if args.net_type == 'ps':
        helper = PSHelper(args, helper_args, config)
    elif args.net_type == 'mvs':
        helper = MVSHelper(args, helper_args, config)

    print_peak_memory("Max memory allocated after creating local model", local_rank)
    print('Initialization complete..')

    save_checkpoint = save_checkpoint_func(helper.model, helper.optimizer, helper.scheduler, helper.scaler)

    gc.collect()
    if device_name == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.synchronize()
    losses = {}
    for epoch in np.arange(helper.initial_epoch-1, args.epochs):
        if is_distributed:
            helper.train_sampler.set_epoch(int(epoch))
            helper.val_sampler.set_epoch(int(epoch))

        # train
        print('Training...')
        log_index = (helper.num_batches_train + helper.num_batches_val) * epoch
        losses['train_loss'] = helper.train(logger, log_index, epoch)

        # save snapshot of model
        if is_distributed:
            helper.optimizer.consolidate_state_dict()
        if local_rank == 0 and (epoch + 1) % args.save_freq == 0:
            print('saving checkpoint...')
            save_checkpoint(checkpoint_dir, epoch)
            with open(args.ckpt_to_continue, 'w') as ckpt:
                ckpt.write(str(epoch + 1))

        # validation
        print('Validating...')
        if (epoch + 1) % args.eval_freq == 0 or (epoch + 1) == args.epochs:
            with torch.no_grad():
                log_index = helper.num_batches_train * (epoch + 1) + helper.num_batches_val * epoch
                losses['val_loss'], _ = helper.validate(logger, log_index, epoch)

        # record train and validation loss
        if local_rank == 0:
            logger.add_scalars('Loss', losses, epoch + 1)
            print("Updated Train/Val loss graph...")

    if device_name == 'cuda': torch.cuda.synchronize()

def train_tune(config, checkpoint_dir=None, args=None, helper_args=None):
    time.sleep(90)
    if device_name == 'cuda': tune.utils.wait_for_gpu()

    helper = None
    if args.net_type == 'ps':
        helper = PSHelper(args, helper_args, config)
    elif args.net_type == 'mvs':
        helper = MVSHelper(args, helper_args, config)

    print_peak_memory("Max memory allocated after creating local model", local_rank)

    if checkpoint_dir:
        print(f'Loading model...')
        helper.initial_epoch, optim_state, scheduler_state, scaler_state = load_checkpoint(
            os.path.join(checkpoint_dir, 'checkpoint'), helper.model.module, device=None)

        helper.optimizer.load_state_dict(optim_state)
        helper.scheduler.load_state_dict(scheduler_state)
        helper.scaler.load_state_dict(scaler_state)
        print('Model loaded successfully')


    gc.collect()
    if device_name == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.synchronize()
    for epoch in np.arange(helper.initial_epoch, args.epochs):
        # train
        print('Training...')
        log_index = (helper.num_batches_train + helper.num_batches_val) * epoch
        helper.train(None, log_index, epoch)

        # validation
        print('Validating...')
        if (epoch + 1) % args.eval_freq == 0 or (epoch + 1) == args.epochs:
            with torch.no_grad():
                log_index = helper.num_batches_train * (epoch + 1) + helper.num_batches_val * epoch
                val_loss, info = helper.validate(None, log_index, epoch)

        # save snapshot of model
        if (epoch + 1) % args.save_freq == 0:
            print('saving checkpoint...')
            with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, "checkpoint")
                torch.save({'epoch': epoch + 1,
                            'model': helper.model.module.state_dict(),
                            'optimizer': helper.optimizer.state_dict(),
                            'scheduler': helper.scheduler.state_dict(),
                            'scaler': helper.scaler.state_dict()}, path)

        tune.report(loss=val_loss, info=info)
    if device_name == 'cuda': torch.cuda.synchronize()

if __name__=='__main__':
    main(args)