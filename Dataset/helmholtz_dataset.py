import os, sys
sys.path.append('.')
import numpy as np
from torch.utils.data import Dataset

import Dataset.preprocessing as prep


class Helmholtz_Dataset(Dataset):

    def __init__(self, root_dir, path_to_obj_dir_list, num_reciprocals=4, num_views=8, mode='train', net='mvs', **kwargs):
        super(Helmholtz_Dataset, self).__init__()

        self.root_dir = root_dir
        self.views_data = prep.parse_views(self.root_dir, path_to_obj_dir_list, num_views)
        self.num_reciprocals = num_reciprocals
        self.mode = mode
        self.net = net
        if self.net == 'mvs':
            self.max_stages = 3
            num_stages = kwargs['num_stages']
            self.num_stages = num_stages if num_stages else self.max_stages
            self.scale_factors =[2**(self.max_stages-1) // 2**i for i in range(self.num_stages)]

    def __len__(self):
        return len(self.views_data)

    def get_data(self, path_to_data, prefix, conc_light=False):
        image, projection_mat, light_pos = prep.get_image_data(path_to_data, prefix)
        #image = (image * np.random.uniform(1, 3)).clip(0, 2)
        #image =  prep.randomNoise(image)
        light_pos = np.broadcast_to(light_pos, image.shape)
        image = np.concatenate((image, light_pos), 2) #shape: (H, W, 6)
        if conc_light:
            image = (image * np.random.uniform(1, 3)).clip(0, 2)
            image =  prep.randomNoise(image)
            light_pos = np.broadcast_to(light_pos, image.shape)
            image_light = np.concatenate((image, light_pos), 2) #shape: (H, W, 6)

            return image_light, projection_mat

        return image, projection_mat


    def mvs_get_item(self, data_dir):
        images = []
        original_projection_mats = []
        data = {}

        path_to_depth_range = os.path.join(data_dir, 'depth_range.txt')
        path_to_depth_map = os.path.join(data_dir, 'depth0001.exr')

        image_light, projection_mat = self.get_data(data_dir, "")
        min_depth, max_depth = np.float32(open(path_to_depth_range).readlines()[0].split())
        images.append(image_light)
        original_projection_mats.append(projection_mat)

        data_dir = os.path.join(data_dir, 'reciprocals')
        for i in range(self.num_reciprocals):
            for j in range(2):
                image_light, projection_mat = self.get_data(data_dir, f'{i + 1}_{j + 1}_')

                images.append(image_light)
                original_projection_mats.append(projection_mat)

        original_projection_mats = np.stack(original_projection_mats)
        projection_mats = {}
        for i in range(self.num_stages):
            if i == self.max_stages-1:
                projection_mats[f'stage{i + 1}'] = original_projection_mats
            else:
                scale_factor = self.scale_factors[i]
                curr_stage_projection_mats = original_projection_mats.copy()
                curr_stage_projection_mats[:, 1, :2, :3] = original_projection_mats[:, 1, :2, :3] /scale_factor
                projection_mats[f'stage{i + 1}'] = curr_stage_projection_mats

        #stage2_projection_mats = stage3_projection_mats.copy()
        #stage2_projection_mats[:, 1, :2, :3] = stage3_projection_mats[:, 1, :2, :3] / 2.
        #stage1_projection_mats = stage3_projection_mats.copy()
        #stage1_projection_mats[:, 1, :2, :3] = stage3_projection_mats[:, 1, :2, :3] / 4.
        #projection_mats = {'stage1': stage1_projection_mats, 'stage2': stage2_projection_mats,
         #                  'stage3': stage3_projection_mats}

        images = np.stack(images).transpose([0, 3, 1, 2])

        data['imgs'] = images
        data['projection_mats'] = projection_mats
        data['depth_values'] = np.array([min_depth, max_depth], np.float32)

        if self.mode == 'train':
            depth_maps = prep.load_depth_maps(path_to_depth_map, self.scale_factors, self.max_stages)
            masks = prep.generate_masks(depth_maps, min_depth, max_depth)

            data['depth_gts'] = depth_maps
            data['masks'] = masks

        return data

    def ps_get_item(self, data_dir):
        data = {}
        images = []
        projection_mats = []

        path_to_mask = os.path.join(data_dir, 'mask0001.exr')
        path_to_normal = os.path.join(data_dir, 'normal0001.exr')

        image_light, projection_mat = self.get_data(data_dir, '', conc_light=True)
        images.append(image_light)
        projection_mats.append(projection_mat)

        data_dir = os.path.join(data_dir, 'reciprocals')
        for i in range(self.num_reciprocals):
            for j in range(2):
                image_light, projection_mat = self.get_data(data_dir, f'{i + 1}_{j + 1}_', conc_light=True)

                images.append(image_light)
                projection_mats.append(projection_mat)

        mask = prep.read_exr_image(path_to_mask)
        mask[mask != 1.0] = 0.0

        normal = prep.read_exr_image(path_to_normal)
        norm = np.sqrt((normal * normal).sum(2, keepdims=True))
        normal = normal / (norm + 1e-12)


        data['mask'] = mask[:, :, 1]
        data['normal_gt'] = normal.transpose([2, 0, 1])

        images = np.stack(images).transpose([0, 3, 1, 2]) #shape: (num_reciprocals + 1, 6, H, W)
        projection_mats = np.stack(projection_mats)

        data['imgs'] = images
        data['projection_mats'] = projection_mats

        return data

    def __getitem__(self, idx):
        view = self.views_data[idx]
        data_dir = os.path.join(self.root_dir, view)

        data = None
        if self.net == 'mvs':
            data = self.mvs_get_item(data_dir)
        elif self.net == 'ps':
            data = self.ps_get_item(data_dir)

        data['view'] = view

        return data


# For testing purposes
if __name__ == '__main__':
    import cv2
    train_list = '/Users/culsu/Documents/UNI_stuff/Surrey/Courses/MSC_Project/src/downloads/' \
                        'Helmholtz_dataset/ShapeNetSem/train.txt'
    root_dir = '/Users/culsu/Documents/UNI_stuff/Surrey/Courses/MSC_Project/src/downloads/' \
                        'Helmholtz_dataset/'
    dataset = Helmholtz_Dataset(root_dir, train_list)
    for index in range(len(dataset.views_data)):
        some_data = dataset.__getitem__(index)
        print(some_data['imgs'].shape)
        print(some_data['normal_gt'].shape)
        print(some_data['view'])
        some_data['imgs'] = some_data['imgs'].transpose([0, 2, 3, 1])
        some_data['normal_gt'] = some_data['normal_gt'].transpose([1, 2, 0])
        print([[f'stage{id +1}', some_data['projection_mats'][f'stage{id + 1}'].shape] for id in range(len(some_data['projection_mats']))])
        print(some_data['depth_values'])

        [prep.imshow_exr(f'test_image{id}',some_data['imgs'][id, :,:,:3]) for id in range(some_data['imgs'].shape[0])]

        prep.imshow_exr(f'test_normal{id}', some_data['normal_gt'])

        [prep.imshow_exr(f'test_depth{id}', some_data['depth_gts'][f'stage{id + 1}']) for id in range(len(some_data['depth_gts']))]

        [prep.imshow_exr(f'test_mask{id}', some_data['masks'][f'stage{id + 1}']) for id in range(len(some_data['masks']))]

        cv2.waitKey()
