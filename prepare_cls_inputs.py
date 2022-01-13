import numpy as np
from PIL import Image
import os
import argparse
from tqdm import tqdm
from collections import Counter
from utils.util import crop_validation_images
import png

def online_cut_patches(im, im_size=96, stride=32):
    """
    function for crop the image to subpatches, will include corner cases
    the return position (x,y) is the up left corner of the image
    Args:
        im (np.ndarray): the image for cropping
        im_size (int, optional): the sub-image size. Defaults to 56.
        stride (int, optional): the pixels between two sub-images. Defaults to 28.
    Returns:
        (list, list): list of image reference and list of its corresponding positions
    """
    im_list = []
    position_list = []

    h, w, _ = im.shape
    if h < im_size:
        h_ = np.array([0])
    else:
        h_ = np.arange(0, h - im_size + 1, stride)
        if h % stride != 0:
            h_ = np.append(h_, h-im_size)

    if w < im_size:
        w_ = np.array([0])
    else:
        w_ = np.arange(0, w - im_size + 1, stride)
        if w % stride != 0:
            w_ = np.append(w_, w - im_size)

    for i in h_:
        for j in w_:   	
            temp = np.uint8(im[i:i+im_size,j:j+im_size,:])
            im_list.append(temp)
            position_list.append((i,j))
    return im_list, position_list

def prepare_warwick(side_length: int, stride: int) -> None:
    """
    crop the training images and rename it with project convention
    e.g. imageName-[tumor, stroma, normal].png
    the image will be resized to (224, 224)

    Args:
        side_length (int): the crop image length
        stride (int): the steps for cutting a new image
    """
    
    image_path = 'Dataset_warwick/1.training/origin_ims'
    mask_path = 'Dataset_warwick/1.training/mask'
    destination = 'Dataset_warwick/1.training/img' # the output directory of the cropped images
    if not os.path.exists(destination):
        os.mkdir(destination)
    
    training_ims = os.listdir(image_path)
    summary = []

    print(f'start generating train images ...')
    for i in tqdm(range(1, len(training_ims)+1)):
        image_name = f'train_{i}.bmp'
        mask_name = f'train_{i}_anno.bmp'
        im = np.asarray(Image.open(os.path.join(image_path, image_name)))
        mask = np.asarray(Image.open(os.path.join(mask_path, mask_name)))
        crop_list, positions_list = online_cut_patches(im, side_length, stride)
        for j in range(len(crop_list)):
            crop_im = crop_list[j]
            position = positions_list[j]
            crop_mask = mask[position[0]:position[0]+side_length, position[1]:position[1]+side_length]
            has_tumor = 0
            has_normal = 0
            if np.sum(crop_mask > 0) / crop_mask.size > 0.05:
                has_tumor = 1
            if np.sum(crop_mask == 0) / crop_mask.size > 0.05:
                has_normal = 1
            Image.fromarray(crop_im).resize((224,224)).save(os.path.join(destination, f'train_{i}_{j}-[{has_tumor}, {has_normal}].png'))
            summary.append((has_tumor, has_normal))
    
    print(Counter(summary))
    print('finish processing training images!')
    print()
    print('start processing validation and test images...')

    validation_cam_folder_name = 'warwick_valid_out_cam'
    validation_dataset_path = 'Dataset_warwick/2.validation/img'
    scales = [1, 1.25, 1.5, 1.75, 2]
    if not os.path.exists(validation_cam_folder_name):
        os.mkdir(validation_cam_folder_name)

    print('crop validation set images ...')
    crop_validation_images(validation_dataset_path, 224, int(224//3), scales, validation_cam_folder_name)
    print('cropping finishes!')

    def process_mask(mask_folder_path, destination):
        for mask_name in os.listdir(mask_folder_path):
            mask = np.asarray(Image.open(os.path.join(mask_folder_path, mask_name))).copy()
            # this three steps, convert tumor to 0, background to 2
            mask[mask > 0] = 3
            mask[mask == 0] = 1
            mask[mask == 3] = 0
            palette = [(0, 64, 128), (64, 128, 0), (243, 152, 0), (255, 255, 255)]
            with open(os.path.join(destination, f'{mask_name.split(".")[0]}.png'), 'wb') as f:
                w = png.Writer(mask.shape[1], mask.shape[0],palette=palette, bitdepth=8)
                w.write(f, mask.astype(np.uint8))

    validation_mask_path = 'Dataset_warwick/2.validation/origin_mask'
    destination = 'Dataset_warwick/2.validation/mask'
    if not os.path.exists(destination):
        os.mkdir(destination)
    process_mask(validation_mask_path, destination)

    test_mask_path = 'Dataset_warwick/3.testing/origin_mask'
    destination = 'Dataset_warwick/3.testing/mask'
    if not os.path.exists(destination):
        os.mkdir(destination)
    process_mask(test_mask_path, destination)
    print('mask processing finished!')

def prepare_crag(side_length: int, stride: int) -> None:
    """
    crop the training images and rename it with project convention
    e.g. imageName-[tumor, stroma, normal].png
    the image will be resized to (224, 224)

    Args:
        side_length (int): the crop image length
        stride (int): the steps for cutting a new image
    """
    
    image_path = 'Dataset_crag/1.training/origin_ims'
    mask_path = 'Dataset_crag/1.training/mask'
    destination = 'Dataset_crag/1.training/img' # the output directory of the cropped images
    if not os.path.exists(destination):
        os.mkdir(destination)
    
    training_ims = os.listdir(image_path)
    summary = []

    print(f'start generating train images ...')
    for i in tqdm(range(1, len(training_ims)+1)):
        image_name = f'train_{i}.png'
        mask_name = f'train_{i}.png'
        im = np.asarray(Image.open(os.path.join(image_path, image_name)))
        mask = np.asarray(Image.open(os.path.join(mask_path, mask_name)))
        crop_list, positions_list = online_cut_patches(im, side_length, stride)
        for j in range(len(crop_list)):
            crop_im = crop_list[j]
            position = positions_list[j]
            crop_mask = mask[position[0]:position[0]+side_length, position[1]:position[1]+side_length]
            has_tumor = 0
            has_normal = 0
            if np.sum(crop_mask > 0) / crop_mask.size > 0.05:
                has_tumor = 1
            if np.sum(crop_mask == 0) / crop_mask.size > 0.05:
                has_normal = 1
            Image.fromarray(crop_im).save(os.path.join(destination, f'train_{i}_{j}-[{has_tumor}, {has_normal}].png'))
            summary.append((has_tumor, has_normal))
    
    print(Counter(summary))
    print('finish processing training images!')
    print()
    print('start processing validation and test images...')

    validation_cam_folder_name = 'crag_valid_out_cam'
    validation_dataset_path = 'Dataset_crag/2.validation/img'
    scales = [1, 1.25, 1.5, 1.75, 2]
    if not os.path.exists(validation_cam_folder_name):
        os.mkdir(validation_cam_folder_name)

    print('crop validation set images ...')
    crop_validation_images(validation_dataset_path, 224, int(224//3), scales, validation_cam_folder_name)
    print('cropping finishes!')

    def process_mask(mask_folder_path, destination):
        for mask_name in os.listdir(mask_folder_path):
            mask = np.asarray(Image.open(os.path.join(mask_folder_path, mask_name))).copy()
            # this three steps, convert tumor to 0, background to 2
            mask[mask > 0] = 3
            mask[mask == 0] = 1
            mask[mask == 3] = 0
            palette = [(0, 64, 128), (64, 128, 0), (243, 152, 0), (255, 255, 255)]
            with open(os.path.join(destination, f'{mask_name.split(".")[0]}.png'), 'wb') as f:
                w = png.Writer(mask.shape[1], mask.shape[0],palette=palette, bitdepth=8)
                w.write(f, mask.astype(np.uint8))

    validation_mask_path = 'Dataset_crag/2.validation/origin_mask'
    destination = 'Dataset_crag/2.validation/mask'
    if not os.path.exists(destination):
        os.mkdir(destination)
    process_mask(validation_mask_path, destination)

    test_mask_path = 'Dataset_crag/3.testing/origin_mask'
    destination = 'Dataset_crag/3.testing/mask'
    if not os.path.exists(destination):
        os.mkdir(destination)
    process_mask(test_mask_path, destination)
    print('mask processing finished!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', default='warwick', type=str, choices=['warwick', 'wsss', 'crag'], help='now only support three types: (warwick, wsss)')
    parser.add_argument('-side_length', default=112, type=int)
    parser.add_argument('-stride', default=56, type=int)
    args = parser.parse_args()
    
    dataset = args.dataset
    side_length = args.side_length
    stride = args.stride

    if dataset == 'warwick':
        prepare_warwick(side_length, stride)
    
    if dataset == 'crag':
        prepare_crag(side_length, stride)
