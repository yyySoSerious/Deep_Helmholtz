import os, random, glob, cv2
import numpy as np
import torch.nn.functional

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"


def makedir(path:str):
    if not os.path.exists(path):
        os.makedirs(path)


# test ratio is implied
def split_dataset(path_to_dataset_list:str, output_dir:str, train_ratio=0.70, val_ratio=0.20):
    makedir(output_dir)
    train_path = os.path.join(output_dir, 'train.txt')
    val_path = os.path.join(output_dir, 'val.txt')
    test_path = os.path.join(output_dir, 'test.txt')

    # create dataset list
    dataset_list = open(path_to_dataset_list).readlines()
    dataset_list = list(map(lambda path: path.strip(), dataset_list))[1:]
    num_objs = len(dataset_list)

    # shuffle list and split them according train ratio and val ratio
    random.shuffle(dataset_list)
    train_size = round(train_ratio * num_objs)
    val_size = round(val_ratio * num_objs)
    test_size = num_objs - (train_size + val_size)
    print(f'Number of objects: {num_objs}')
    print(f'Number of train objects: {train_size}')
    print(f'Number of validation objects: {val_size}')
    print(f'Number of test objects: {test_size}')

    # save the splits in separate files
    with open(train_path, 'w') as train_f, open(val_path, 'w') as val_f, open(test_path, 'w') as test_f:
        train_f.write(str(train_size) + '\n')
        train_f.write('\n'.join(str(path) for path in dataset_list[:train_size]))

        val_end = train_size + val_size
        val_f.write(str(val_size) + '\n')
        val_f.write('\n'.join(str(path) for path in dataset_list[train_size:val_end]))

        test_f.write(str(test_size) + '\n')
        test_f.write('\n'.join(str(path) for path in dataset_list[val_end:]))


def parse_cameras(camera_dir: str, prefix: str):
    extrinsic_path = os.path.join(camera_dir, f'{prefix}_RT_matrix.txt')
    intrinsic_path = os.path.join(camera_dir, f'{prefix}_K_matrix.txt')
    extrinsic_mat = np.loadtxt(extrinsic_path)
    extrinsic_mat = np.concatenate((extrinsic_mat, np.array([[0.0, 0.0, 0.0, 1.0]])))
    intrinsic_mat = np.loadtxt(intrinsic_path)

    return extrinsic_mat, intrinsic_mat


def save_camera(save_path, projection_mat):
    makedir(save_path)
    extrinsic_mat = projection_mat[0, :4, :4]
    intrinsic_mat = projection_mat[1, :3, :3]
    np.savetxt(os.path.join(save_path, 'K_matrix.txt'), np.matrix(intrinsic_mat))
    np.savetxt(os.path.join(save_path, 'RT_matrix.txt'), np.matrix(extrinsic_mat))


def read_exr_image(path_to_image:str):
    return cv2.imread(
        path_to_image, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)


def normalize_exr_image(image: np.ndarray):
    image = image - image.min()
    return image / (image.max() +1e-12)


def imshow_exr(window_name: str, image: np.ndarray):
    cv2.imshow(window_name, np.uint8(normalize_exr_image(image)*255))

def parse_objs(root_dir:str, path_to_obj_dir_list:str, num_sel_views):
    path_to_obj_dir_list = os.path.join(root_dir, path_to_obj_dir_list)
    obj_dirs = open(path_to_obj_dir_list).readlines()
    obj_dirs = list(map(lambda path: path.strip(), obj_dirs))
    obj_dirs = obj_dirs[1:]
    views_data = []
    for obj_dir in obj_dirs:
        path_to_pair = os.path.join(root_dir, obj_dir, 'pair.txt')
        views_list = open(path_to_pair).readlines()
        view_selections = list(map(lambda path: path.strip(), views_list))
        num_views = int(view_selections[0])
        view_selections = view_selections[1:]
        for i in range(num_views):
            ref_id = view_selections[i * 2]
            src_data = view_selections[i * 2 + 1].split()

            src_ids = [src_data[j * 2 + 1] for j in range(num_sel_views)]
            views_data.append({'obj_dir': obj_dir, 'image_ids': [ref_id] + src_ids})

    return views_data


def parse_views(root_dir:str, path_to_obj_dir_list:str, num_views=8):
    path_to_obj_dir_list = os.path.join(root_dir, path_to_obj_dir_list)
    obj_dirs = open(path_to_obj_dir_list).readlines()
    obj_dirs = list(map(lambda path: path.strip(), obj_dirs))[1:]

    views_data = []
    for obj_dir in obj_dirs:
        for i in range(num_views):
            views_data.append(os.path.join(obj_dir, f'view_{i + 1}'))

    return views_data


def load_depth_maps(path_to_depth_map:str, scale_factors, max_stages):
    num_stages = len(scale_factors)
    original_depth_map = read_exr_image(path_to_depth_map)[:,:, 0]
    h, w = original_depth_map.shape
    depth_maps = {}
    #depth_maps[f'stage{num_stages}'] = original_depth_map
    for i in range(num_stages):
        if i == max_stages - 1:
            depth_maps[f'stage{i + 1}'] = original_depth_map
        else:
            scale_factor = scale_factors[i]
            depth_maps[f'stage{i + 1}'] = cv2.resize(original_depth_map, (w//scale_factor, h//scale_factor),
                                                     interpolation=cv2.INTER_NEAREST)

    #stage2_depth_map = cv2.resize(stage3_depth_map, (w//2, h//2), interpolation=cv2.INTER_NEAREST)
    #stage1_depth_map = cv2.resize(stage3_depth_map, (w//4, h//4), interpolation=cv2.INTER_NEAREST)
    return depth_maps #{'stage1': stage1_depth_map, 'stage2': stage2_depth_map, 'stage3': stage3_depth_map}

def get_image_data(data_dir: str, prefix: str):
    path_to_image = os.path.join(data_dir, f'{prefix}0001.exr')
    image = normalize_exr_image(read_exr_image(path_to_image))

    extrinsic_mat, intrinsic_mat = parse_cameras(data_dir, prefix)

    projection_mat = np.zeros((2, 4, 4), np.float32)
    projection_mat[0, :4, :4] = extrinsic_mat
    projection_mat[1, :3, :3] = intrinsic_mat

    light_pos = np.loadtxt(os.path.join(data_dir, f'{prefix}_light_position.txt'), dtype='float32')
    light_pos = light_pos/(np.linalg.norm(light_pos) +1e-12)

    return image, projection_mat, light_pos

def generate_masks(depth_maps:dict, min_depth, max_depth):
    masks = {}
    for stage, depth_map in depth_maps.items():
        mask = np.ones(depth_map.shape, np.uint8)
        mask[depth_map > max_depth] = 0
        mask[depth_map < min_depth] = 0
        masks[stage] = mask
    return masks

def randomNoise(image, factor=0.05):
    noise = np.random.random(image.shape)
    image += image + (noise - 0.5) * factor

    return image


# For testing purposes
if __name__ == '__main__':
    root_dir = '../..'
    #path_to_objs_list = '/Users/culsu/Documents/UNI_stuff/Surrey/Courses/MSC_Project/src/Helmholtz_dataset/obj_dirs.txt'
    #save_path = '/Users/culsu/Documents/UNI_stuff/Surrey/Courses/MSC_Project/src/test'
    #split_dataset(path_to_objs_list, save_path, train_ratio=0.01, val_ratio=0.01)
    #split_dataset('../../Helmholtz_dataset/obj_dirs.txt', '.')

    #extrinsic, intrinsic = parse_cameras('../../Helmholtz_dataset/03325088/1a5586fc147de3214b35a7d7cea7130/view_2/right_reciprocal')
    #print('extrinsic matrix', extrinsic, 'shape:', extrinsic.shape)
    #print('intrinsic matrix', intrinsic, 'shape:', intrinsic.shape)

    #image = read_exr_image("/Users/culsu/Documents/UNI_stuff/Surrey/Courses/MSC_Project/src/Helmholtz_dataset/TissueBox/a65eb3d3898cdd9a48e2056fc010654d/view_6/normal0001.exr")

    #tensor_img = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0)
    #if torch.isnan(tensor_img).any():
     #   print('im nan bitch')
    #else:
     #   print('no nan :(')
    #print(torch.isnan(tensor_img))
    #print('the size of this tensor image is: ', tensor_img.shape)
    #this = torch.nn.functional.interpolate(tensor_img, [800, 1500]).squeeze(0)
    #print(this.shape)
    #image = this.numpy()
    #image = image.transpose(1, 2, 0)
    #imshow_exr("imae", image)
    #cv2.waitKey()
