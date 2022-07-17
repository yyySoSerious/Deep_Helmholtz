import os, time, cv2, gc
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import numpy as np

import flags
from utils.helper_funcs import *
from networks.helmholtz_network import HelmholtzNet
from Dataset.helmholtz_dataset import Helmholtz_Dataset
from Dataset.preprocessing import makedir, normalize_exr_image, save_camera

args = flags.TestFlags().parse()

#TODO: change to 'mps:0' when conv3d and other relevant functions have been implemented for mps
device_name = 'cuda:0' if torch.cuda.is_available() else 'cpu' if torch.backends.mps.is_available() else 'cpu'
device = torch.device(device_name)


def initialize():
    test_set = Helmholtz_Dataset(root_dir=args.root_dir, path_to_obj_dir_list=args.test_list,
                                  num_sel_views=args.num_sel_views, mode='test')
    test_loader = torch.utils.data.DataLoader(test_set, 1, num_workers=4, drop_last=False, shuffle=False,
                                              pin_memory=device_name != 'cpu')

    model = HelmholtzNet(planes_in_stages=list(map(int, args.planes_in_stages.split(','))),
                         conf_lambda=args.conf_lambda)
    print(f'Loading  model {args.ckpt} ...')
    load_checkpoint(args.ckpt, model)
    print(f'Model loaded successfully')
    model.to(device)
    model = torch.nn.DataParallel(model)

    return model, test_loader


def main(args, model, test_loader):
    reciprocals = {'left': 'left_reciprocal',
                        'right': 'right_reciprocal'}
    model.eval()
    total_time = 0

    for batch_idx, sample in enumerate(test_loader):
        obj_dir = sample['view']['obj_dir'][0]
        view_id, reciprocal_id = sample['view']['image_ids'][0][0].split('_')
        data_dir = os.path.join(args.save_dir, obj_dir, f'view_{view_id}',
                                reciprocals[reciprocal_id])
        makedir(data_dir)

        sample_device = dict_to_device(sample, device)

        print(f'Testing {data_dir} ...')
        initial_time = time.time()
        outputs = model(sample_device['imgs'], sample_device['projection_mats'], sample_device['depth_values'])
        end_time = time.time()

        outputs = dict_to_numpy(outputs)
        del sample_device

        total_time += (end_time - initial_time)

        print('Finished {}/{}, time: {:.2f}s (Total time: {:.2f}s).'.format(batch_idx + 1, len(test_loader),
                                                                      end_time - initial_time,
                                                                      total_time / (batch_idx + 1.)))


        #todo: adjust this if the images were normalized to [0 1]
        ref_img = sample['imgs'][0] [0].numpy().transpose(1, 2, 0) #* 255
        ref_img = np.uint8(normalize_exr_image(ref_img)*255)
        cv2.imwrite(f'{data_dir}/{reciprocals[reciprocal_id]}.png', ref_img)

        camera_params = sample['projection_mats']['stage3'][0, 0].numpy()
        save_camera(data_dir, camera_params)

        for stage_idx in range(3):
            curr_stage = outputs[f'stage{stage_idx+1}']
            curr_depth = curr_stage['depth'][0]
            curr_confidence = curr_stage['confidence'][0]
            cv2.imwrite(data_dir + f'/depth_{stage_idx+1}.exr', curr_depth)
            cv2.imwrite(data_dir + f'/confidence_{stage_idx+1}.exr', curr_confidence)

        print('rgb, depth and confidence images saved for this view point')

    if device_name == 'cuda:0': torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == '__main__':
    model, test_loader = initialize()
    with torch.no_grad():
        main(args, model, test_loader)