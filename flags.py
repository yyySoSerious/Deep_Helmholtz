import argparse

class TrainFlags:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Training Helmholtz Network")
        self.setup()

    def setup(self):
        # File paths Arguments
        self.parser.add_argument('--root_dir', type=str, help='path to root directory')
        self.parser.add_argument('--train_list', type=str, help='Path to list of train objects',
                            default='Deep_Helmholtz/Dataset/train.txt')
        self.parser.add_argument('--val_list', type=str, help='Path to list of validation objects',
                                 default='Deep_Helmholtz/Dataset/val.txt')
        self.parser.add_argument('--save_dir', type=str, help='path to save checkpoints')

        # Training Arguments
        self.parser.add_argument('--ckpt_to_continue', type=str, default=None)
        self.parser.add_argument('--epochs', type=int, default=60)
        self.parser.add_argument('--lr', type=float, default=0.0016)
        self.parser.add_argument('--lr_decay', type=float, default=0.0, help='weight decay')
        self.parser.add_argument('--lr_idx', type=str, default="10,12,14:0.5")
        self.parser.add_argument('--beta_1', type=float, default=0.9, help='adam')
        self.parser.add_argument('--beta_2', type=float, default=0.999, help='adam')
        self.parser.add_argument('--loss_weights', type=str, default="0.5,1.0,2.0")
        self.parser.add_argument('--batch', type=int, default=1)

        # ucs args
        self.parser.add_argument('--num_sel_views', type=int, help='num of candidate views', default=1)
        self.parser.add_argument('--conf_lambda', type=float, help='the interval coefficient', default=1.5)
        self.parser.add_argument('--planes_in_stages', type=str, help='number of samples for each stage.', default='64,32,8')

        self.parser.add_argument('--log_freq', type=int, default=50, help='print and summary frequency')
        self.parser.add_argument('--save_freq', type=int, default=1, help='save checkpoint frequency.')
        self. parser.add_argument('--eval_freq', type=int, default=1, help='evaluate frequency.')

        self.parser.add_argument('--sync_bn', action='store_true', help='Sync BN.')
        self.parser.add_argument('--opt_level', type=str, default="O0")
        self.parser.add_argument('--seed', type=int, default=0)
        self.parser.add_argument('--system', type=str, default='LINUX')


        self.parser.add_argument('--net_type', type=str, help='the type of network to train', default='helmholtz')

    def parse(self):
        self.args = self.parser.parse_args()
        return self.args


class TestFlags:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Testing Helmholtz Network")
        self.setup()

    def setup(self):
        # File paths Arguments
        self.parser.add_argument('--root_dir', type=str, help='path to root directory')
        self.parser.add_argument('--test_list', type=str, help='Path to list of train objects',
                                 default='Deep_Helmholtz/Dataset/test.txt')
        self.parser.add_argument('--save_dir', type=str, help='path to save depth maps')

        # Testing Arguments
        self.parser.add_argument('--ckpt', default=None)
        self.parser.add_argument('--num_sel_views', type=int, help='num of candidate views', default=1)
        self.parser.add_argument('--conf_lambda', type=float, help='the interval coefficient', default=1.5)
        self.parser.add_argument('--planes_in_stages', type=str, help='number of samples for each stage.',
                                 default='64,32,8')


        self.parser.add_argument('--net_type', type=str, help='the type of network to train', default='helmholtz')


    def parse(self):
        self.args = self.parser.parse_args()
        return self.args