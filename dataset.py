from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image
import torch
from utils.util import online_cut_patches
class OriginPatchesDataset(Dataset):
    def __init__(self, data_path_name = "Dataset/1.training", transform=None):
        self.path = data_path_name
        self.files = os.listdir(data_path_name)
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.path, self.files[idx])
        im = Image.open(image_path)

        if self.transform:
            im = self.transform(im)
        label = np.array([int(self.files[idx][-12]), int(self.files[idx][-9]), int(self.files[idx][-6])])
        return im, label

# class OnlineDataset(Dataset):
#     def __init__(self, data_path_name, transform=None, patch_size = 56, stride=28):
#         self.path = data_path_name
#         self.files = os.listdir(data_path_name)
#         self.transform = transform
#         self.patch_size = patch_size
#         self.stride = stride

#     def __len__(self):
#         return len(self.files)

#     def __getitem__(self, idx):
#         image_path = os.path.join(self.path, self.files[idx])
#         im = np.asarray(Image.open(image_path))
#         im_list, position_list = online_cut_patches(im, self.patch_size, self.stride)
#         if self.transform:
#             for patch_id in range(len(im_list)):
#                 im_list[patch_id] = self.transform(im_list[patch_id])
#         # label = int(self.files[idx][-5:-4])
#         # position = tuple(int(self.files[idx][-7]), int(self.files[idx][-8]))
#         return image_path, im_list, position_list

# class OnlineTrainDataset(Dataset):
#     def __init__(self, data_path_name, transform=None):
#         self.path = data_path_name
#         self.files = os.listdir(data_path_name)
#         self.transform = transform

#     def __len__(self):
#         return len(self.files)

#     def __getitem__(self, idx):
#         image_path = os.path.join(self.path, self.files[idx])
#         im = Image.open(image_path)
#         if self.transform:
#             im = self.transform(im)

#         return image_path, im 
