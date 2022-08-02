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
    dataset_list = list(map(lambda path: path.strip(), dataset_list))
    num_objs = int(dataset_list[0])

    # shuffle list and split them according train ratio and val ratio
    dataset_list = dataset_list[1:]
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
        print(val_end)
        val_f.write(str(val_size) + '\n')
        val_f.write('\n'.join(str(path) for path in dataset_list[train_size:val_end]))

        test_f.write(str(test_size) + '\n')
        test_f.write('\n'.join(str(path) for path in dataset_list[val_end:]))


def parse_cameras(camera_dir:str):
    extrinsic_path = glob.glob(os.path.join(camera_dir, '*_RT_matrix.txt'))
    intrinsic_path = glob.glob(os.path.join(camera_dir, '*_K_matrix.txt'))
    extrinsic_mat = np.loadtxt(extrinsic_path[0])
    extrinsic_mat = np.concatenate((extrinsic_mat, np.array([[0.0, 0.0, 0.0, 1.0]])))
    intrinsic_mat = np.loadtxt(intrinsic_path[0])

    return extrinsic_mat, intrinsic_mat

def save_camera(save_path, projection_mat):
    makedir(save_path)
    extrinsic_mat = projection_mat[0, :4, :4]
    intrinsic_mat = projection_mat[1, :3, :3]
    np.savetxt(save_path + '/K_matrix.txt', np.matrix(intrinsic_mat))
    np.savetxt(save_path + '/RT_matrix.txt', np.matrix(extrinsic_mat))



def read_exr_image(path_to_image:str):
    return cv2.imread(
        path_to_image, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)


def normalize_exr_image(image: str):
    image = image - image.min()
    return image / image.max()


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

def load_depth_maps(path_to_depth_map:str):
    stage3_depth_map = read_exr_image(path_to_depth_map)[:,:, 0]
    h, w = stage3_depth_map.shape
    stage2_depth_map = cv2.resize(stage3_depth_map, (w//2, h//2), interpolation=cv2.INTER_NEAREST)
    stage1_depth_map = cv2.resize(stage3_depth_map, (w//4, h//4), interpolation=cv2.INTER_NEAREST)
    return {'stage1': stage1_depth_map, 'stage2': stage2_depth_map, 'stage3': stage3_depth_map}

def get_image_data(data_dir: str, reciprocal_id: str):
    path_to_image = glob.glob(os.path.join(data_dir, '*_reciprocal0001.exr'))[0]
    image = normalize_exr_image(read_exr_image(path_to_image))


    path_to_normal = glob.glob(os.path.join(data_dir, '*_reciprocal_normal0001.exr'))[0]
    normal = read_exr_image(path_to_normal)

    extrinsic_mat, intrinsic_mat = parse_cameras(data_dir)
    path_to_depth_range = glob.glob(os.path.join(data_dir, '*_depth_range.txt'))[0]
    min_depth, max_depth = open(path_to_depth_range).readlines()[0].split()

    projection_mat = np.zeros((2, 4, 4), np.float32)
    projection_mat[0, :4, :4] = extrinsic_mat
    projection_mat[1, :3, :3] = intrinsic_mat

    light_pos = np.loadtxt(glob.glob(os.path.join(data_dir, '*_light_position.txt'))[0],delimiter=',', dtype="float32")

    return image, normal, np.float32(min_depth), np.float32(max_depth), projection_mat, light_pos


def generate_masks(depth_maps:dict, min_depth, max_depth):
    masks = {}
    for stage, depth_map in depth_maps.items():
        mask = np.ones(depth_map.shape, np.uint8)
        mask[depth_map > max_depth] = 0
        mask[depth_map < min_depth] = 0
        masks[stage] = mask
    return masks


# For testing purposes
if __name__ == '__main__':
    root_dir = '../..'

    #split_dataset('../../Helmholtz_dataset/obj_dirs.txt', '.')

    #extrinsic, intrinsic = parse_cameras('../../Helmholtz_dataset/03325088/1a5586fc147de3214b35a7d7cea7130/view_2/right_reciprocal')
    #print('extrinsic matrix', extrinsic, 'shape:', extrinsic.shape)
    #print('intrinsic matrix', intrinsic, 'shape:', intrinsic.shape)

    image = read_exr_image("/Users/culsu/Documents/UNI_stuff/Surrey/Courses/MSC_Project/src/Build_helmholtz_v2/tmp/view_8/0001.exr")
    #tensor_img = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0)
    #print('the size of this tensor image is: ', tensor_img.shape)
    #this = torch.nn.functional.interpolate(tensor_img, [800, 1500]).squeeze(0)
    #print(this.shape)
    #image = this.numpy()
    #image = image.transpose(1, 2, 0)
    imshow_exr("imae", image)
    cv2.waitKey()
