from Dataset.helmholtz_dataset import Helmholtz_Dataset


class Test:
    def __init__(self, args, **kwargs):
        self.args = args
        test_set = Helmholtz_Dataset(root_dir=self.args.root_dir, path_to_obj_dir_list=self.args.train_list,
                                      num_reciprocals=self.args.num_reciprocals, net=self.args.net_type, mode='test', **kwargs)
