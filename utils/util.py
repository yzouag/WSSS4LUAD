from shutil import copyfile
import shutil
import numpy as np
import torch
from PIL import Image
import os
from collections import Counter
from os.path import join as osp
from tqdm import tqdm


def convertinttoonehot(nums_list: torch.tensor):
    dic = {0: [1, 0, 0], 1: [0, 1, 0], 2: [0, 0, 1]}
    result = np.empty((len(nums_list), 3))
    for i in range(len(nums_list)):
        result[i] = np.array(dic[nums_list[i].item()])

    return torch.tensor(result, requires_grad=False)


def sample_single_label(single_path, sample_size, result_path="sample_single_patches"):
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    else:
        shutil.rmtree(result_path)
        os.mkdir(result_path)
    dic = {0: [], 1: [], 2: []}
    for file in os.listdir(single_path):
        # copyfile(osp(single_path, file), osp(result_path, file))
        index = int(file[-5])
        dic[index].append(file)

    minlength = min(len(dic[0]), len(dic[1]), len(dic[2]))
    select_index = np.random.choice(minlength, sample_size, replace=False)
    for k in tqdm(select_index):
        copyfile(osp(single_path, dic[0][k]), osp(result_path, dic[0][k]))
        copyfile(osp(single_path, dic[1][k]), osp(result_path, dic[1][k]))
        copyfile(osp(single_path, dic[2][k]), osp(result_path, dic[2][k]))
    return


def sample_double_label(double_path, result_path="sample_double_patches"):
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    else:
        shutil.rmtree(result_path)
        os.mkdir(result_path)
    tumor = []
    for file in os.listdir(double_path):
        fileindex = (file[-13: -4])
        if fileindex == "[1, 0, 0]":
            # tumor.append(file)
            continue
        elif fileindex == "[0, 1, 0]" or fileindex == "[1, 1, 0]":
            copyfile(osp(double_path, file), osp(result_path, file))

    # select_index = np.random.choice(len(tumor), 7000, replace=False)
    # for k in tqdm(select_index):
    #     copyfile(osp(double_path, tumor[k]), osp(result_path, tumor[k]))
    return


def calculate_index(path):
    l = []
    for file in os.listdir(path):
        fileindex = (file[-13: -4])
        l.append(fileindex)

    print(Counter(l))
    return


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
            temp = np.uint8(im[i:i+im_size, j:j+im_size, :].copy())
            im_list.append(temp)
    return im_list
