import os
from PIL import Image
import numpy as np
from utils.pyutils import online_cut_patches
import png

# pseudo_mask_path = 'resnet_glas_11256_new_best_train_pseudo_mask'
# origin_ims_path = 'Dataset_warwick/1.training/origin_ims'
# destination = 'glas_ours_trainPseudoMask'

# if not os.path.exists(destination):
#     os.mkdir(destination)

# ims_dict = {}

# for partial_mask in os.listdir(pseudo_mask_path):
#     _, corresponding_im, index = partial_mask.split('_')
#     index = int(index.split('-')[0])
#     if f'train_{corresponding_im}.bmp' not in ims_dict:
#         ims_dict[f'train_{corresponding_im}.bmp'] = {}
#     ims_dict[f'train_{corresponding_im}.bmp'][index] = os.path.join(pseudo_mask_path, partial_mask)


# for origin_im in os.listdir(origin_ims_path):
#     im = np.asarray(Image.open(os.path.join(origin_ims_path, origin_im)))
#     complete_mask = np.zeros((im.shape[0], im.shape[1]))
#     sum_counter = np.zeros_like(complete_mask)
#     _, position_list = online_cut_patches(im, im_size=112, stride=56)
    
#     for i in range(len(position_list)):
#         partial_mask = np.load(ims_dict[origin_im][i], allow_pickle=True)
#         position = position_list[i]
#         complete_mask[position[0]:position[0]+112, position[1]:position[1]+112] += partial_mask
#         sum_counter[position[0]:position[0]+112, position[1]:position[1]+112] += 1

#     complete_mask = np.rint(complete_mask / sum_counter)
#     palette = [(0, 64, 128), (64, 128, 0), (243, 152, 0), (255, 255, 255)]
#     with open(os.path.join(destination, f'{origin_im.split(".")[0]}.png'), 'wb') as f:
#         w = png.Writer(complete_mask.shape[1], complete_mask.shape[0], palette=palette, bitdepth=8)
#         w.write(f, complete_mask.astype(np.uint8))

pseudo_mask_path = 'resnet_glas_11256_new_best_train_pseudo_mask_test'
origin_ims_path = 'Dataset_warwick/2.validation/img'
destination = 'glas_ours_testPseudoMask'

if not os.path.exists(destination):
    os.mkdir(destination)

ims_dict = {}

for partial_mask in os.listdir(pseudo_mask_path):
    _, corresponding_im, index = partial_mask.split('_')
    index = int(index.split('-')[0])
    if f'{corresponding_im}.bmp' not in ims_dict:
        ims_dict[f'{corresponding_im}.bmp'] = {}
    ims_dict[f'{corresponding_im}.bmp'][index] = os.path.join(pseudo_mask_path, partial_mask)


for origin_im in os.listdir(origin_ims_path):
    im = np.asarray(Image.open(os.path.join(origin_ims_path, origin_im)))
    complete_mask = np.zeros((im.shape[0], im.shape[1]))
    sum_counter = np.zeros_like(complete_mask)
    _, position_list = online_cut_patches(im, im_size=224, stride=224)
    
    for i in range(len(position_list)):
        partial_mask = np.load(ims_dict[origin_im][i], allow_pickle=True)
        position = position_list[i]
        complete_mask[position[0]:position[0]+224, position[1]:position[1]+224] += partial_mask
        sum_counter[position[0]:position[0]+224, position[1]:position[1]+224] += 1

    complete_mask = np.rint(complete_mask / sum_counter)
    palette = [(0, 64, 128), (64, 128, 0), (243, 152, 0), (255, 255, 255)]
    with open(os.path.join(destination, f'{origin_im.split(".")[0]}.png'), 'wb') as f:
        w = png.Writer(complete_mask.shape[1], complete_mask.shape[0], palette=palette, bitdepth=8)
        w.write(f, complete_mask.astype(np.uint8))