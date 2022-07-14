import os, glob
import numpy as np

import torch
from torch.utils.data import Dataset

from . import preprocessing as prep

#TODO: send data directly to device
class Helmholtz_Dataset(Dataset):

    def __init__(self, root_dir, path_to_obj_dir_list, num_sel_views=1):
        super(Helmholtz_Dataset, self).__init__()

        self.root_dir = root_dir
        self.views_data = prep.parse_objs(self.root_dir, path_to_obj_dir_list, num_sel_views)
        self.reciprocals = {'left': 'left_reciprocal',
                            'right': 'right_reciprocal'}

    def __len__(self):
        return len(self.views_data)

    def __getitem__(self, idx):
        images = []
        stage3_projection_mats = []
        data = {}

        view = self.views_data[idx]
        image_ids = view['image_ids']
        for i in range(len(image_ids)):
            view_id, reciprocal_id = image_ids[i].split('_')
            data_dir = os.path.join(self.root_dir, view['obj_dir'], f'view_{view_id}',
                                    self.reciprocals[reciprocal_id])
            image, min_depth, max_depth, projection_mat = prep.get_image_data(data_dir)

            images.append(image)
            stage3_projection_mats.append(projection_mat)

            if i == 0:
                depth_maps = prep.load_depth_maps(glob.glob(os.path.join(data_dir, '*_reciprocal_depth0001.exr'))[0])
                masks = prep.generate_masks(depth_maps, min_depth, max_depth)

                data['depth_gts'] = depth_maps
                data['masks'] = masks
                data['depth_values'] = np.array([min_depth, max_depth], np.float32)

                # add reciprocal to the src images
                reciprocal_id = 'right' if reciprocal_id == 'left' else 'left'
                data_dir = os.path.join(self.root_dir, view['obj_dir'], f'view_{view_id}',
                                        self.reciprocals[reciprocal_id])
                image, min_depth, max_depth, projection_mat = prep.get_image_data(data_dir)

                images.append(image)
                stage3_projection_mats.append(projection_mat)

            # print(projection_mat)
            # print(min_depth, max_depth)
            # print(extrinsic_mat, intrinsic_mat)

        stage3_projection_mats = np.stack(stage3_projection_mats)
        stage2_projection_mats = stage3_projection_mats.copy()
        stage2_projection_mats[:, 1, :2, :3] = stage3_projection_mats[:, 1, :2, :3] / 2.
        stage1_projection_mats = stage3_projection_mats.copy()
        stage1_projection_mats[:, 1, :2, :3] = stage3_projection_mats[:, 1, :2, :3] / 4.
        projection_mats = {'stage1': stage1_projection_mats, 'stage2': stage2_projection_mats,
                           'stage3': stage3_projection_mats}

        images = np.stack(images).transpose([0, 3, 1, 2])

        data['imgs'] = images
        data['projection_mats'] = projection_mats
        data['view'] = view

        return data


# For testing purposes
if __name__ == '__main__':
    import cv2
    dataset = Helmholtz_Dataset('../..', 'train.txt', 3)
    for index in range(len(dataset.views_data)):
        some_data = dataset.__getitem__(index)
        print(some_data['imgs'].shape)
        print(some_data['view'])
        some_data['imgs'] = some_data['imgs'].transpose([0, 2, 3, 1])
        print([[f'stage{id +1}', some_data['projection_mats'][f'stage{id + 1}'].shape] for id in range(len(some_data['projection_mats']))])
        print(some_data['depth_values'])
        #[cv2.imshow(f'test{id}',np.uint8(prep.normalize_exr_image(some_data['imgs'][id]))) for id in range(some_data['imgs'].shape[0])]

        #[cv2.imshow(f'test{id}', some_data['depth_maps'][f'stage{id + 1}']) for id in range(len(some_data['depth_maps']))]

        #[cv2.imshow(f'test{id}', some_data['masks'][f'stage{id + 1}']*255) for id in range(len(some_data['masks']))]

        #cv2.waitKey()
