from shutil import copyfile
import shutil
import numpy as np
import torch
from PIL import Image
import os
from patchify import patchify
from collections import Counter
from os.path import join as osp
from tqdm import tqdm

def sample_single_label(single_path, result_path="sample_single_patches"):
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    else:
        shutil.rmtree(result_path)
        os.mkdir(result_path)
    dic = {0 : [], 1 : [], 2 : []}
    for file in os.listdir(single_path):
        index = int(file[-5])
        dic[index].append(file)

    minlength = min(len(dic[0]), len(dic[1]), len(dic[2]))
    select_index = np.random.choice(minlength, 5000, replace=False)
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
        fileindex = (file[-11: -4])
        if fileindex == "[1 0 0]":
            tumor.append(file)
        elif fileindex == "[0 1 0]" or fileindex == "[0 0 1]":
            copyfile(osp(double_path, file), osp(result_path, file))

    select_index = np.random.choice(len(tumor), 3500, replace=False)
    for k in tqdm(select_index):
        copyfile(osp(double_path, tumor[k]), osp(result_path, tumor[k]))
    return

def calculate_index(path):
    l = []
    for file in os.listdir(path):
        fileindex = (file[-11: -4])
        l.append(fileindex)

    print(Counter(l))
    return

if __name__ == "__main__":
    sample_single_label("train_single_patches1")
    sample_double_label("patch5632_train")
    calculate_index("sample_single_patches")
    calculate_index("sample_double_patches")