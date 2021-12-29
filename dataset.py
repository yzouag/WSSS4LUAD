import torch
import re
from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image
from utils.util import multiscale_online_crop
from torchvision import transforms

def get_file_label(filename):
    return np.array([int(filename[-12]), int(filename[-9]), int(filename[-6])])

def find_relative_label(label):
    label_dic = {tuple((1, 0, 0)): [tuple((1, 0, 0)), tuple((1, 1, 0)), tuple((1, 1, 1))],
                tuple((0, 1, 0)): [tuple((1, 1, 0)), tuple((0, 1, 0)), tuple((0, 1, 1))],
                tuple((0, 0, 1)): [tuple((1, 0, 1)), tuple((0, 1, 1)), tuple((0, 0, 1))]}

    return label_dic[label]

class OriginPatchesDataset(Dataset):
    def __init__(self, data_path_name = "Dataset/1.training", transform=None, cutmix_fn=None):
        self.path = data_path_name
        self.files = os.listdir(data_path_name)
        self.transform = transform
        self.filedic = {}
        self.cutmix_fn = cutmix_fn
        self.statedic = {}
        if self.cutmix_fn:
            for filename in self.files:
                filelabel = tuple(get_file_label(filename=filename))
                if filelabel not in self.filedic:
                    self.filedic[filelabel] = [filename]
                else:
                    self.filedic[filelabel].append(filename)
                if filelabel not in self.statedic:
                    self.statedic[filelabel] = 1
        
        self.statedic[tuple((0, 1, 1))] = 1

    def __len__(self):
        return len(self.files)
        # return 50

    def __getitem__(self, idx):
        image_path = os.path.join(self.path, self.files[idx])
        im = Image.open(image_path)
        label = get_file_label(filename=self.files[idx])
        if self.cutmix_fn and label.sum() == 1:
            # choose a single label category based on the current distribution
            relative_labels = find_relative_label(tuple(label))
            current_numbers = np.array([self.statedic[relative_labels[0]], self.statedic[relative_labels[1]], self.statedic[relative_labels[2]]])
            current_probability = [(1/current_numbers[0])/current_numbers.sum(), (1/current_numbers[1])/current_numbers.sum(), (1/current_numbers[2])/current_numbers.sum()]
            activate = np.random.choice([0, 1, 2], p=current_probability)
            mixcategory = np.array((0, 0, 0))
            mixcategory[activate] = 1
            mixcategory = tuple(mixcategory)
            # randomly select a image in that category
            activate = np.random.randint(len(self.filedic[mixcategory]))
            miximage = Image.open(os.path.join(self.path, self.filedic[mixcategory][activate]))
            im = transforms.ToTensor()(im)
            miximage = transforms.ToTensor()(miximage)
            im = self.cutmix_fn(im, miximage, label)
            label = np.logical_or(label, np.array(mixcategory)).astype(np.int32)
            self.statedic[tuple(label)] += 1
        else:
            im = transforms.ToTensor()(im)
            self.statedic[tuple(label)] += 1

        if self.transform:
            im = self.transform(im)
        
        return im, label

class ValidationDataset(Dataset):
    def __init__(self, data_path_name = "Dataset/2.validation/img", transform=None):
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
        return im, self.files[idx]

class OnlineDataset(Dataset):
    def __init__(self, data_path_name, transform, patch_size, stride, scales):
        self.path = data_path_name
        self.files = os.listdir(data_path_name)
        self.transform = transform
        self.patch_size = patch_size
        self.stride = stride
        self.scales = scales

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.path, self.files[idx])
        im = np.asarray(Image.open(image_path))
        scaled_im_list, scaled_position_list = multiscale_online_crop(im, self.patch_size, self.stride, self.scales)
        if self.transform:
            for im_list in scaled_im_list:
                for patch_id in range(len(im_list)):
                    im_list[patch_id] = self.transform(im_list[patch_id])

        return self.files[idx], scaled_im_list, scaled_position_list, self.scales

class OfflineDataset(Dataset):
    def __init__(self, dataset_path, transform):
        self.path = dataset_path
        self.files = os.listdir(self.path)
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.path, self.files[idx])
        im = Image.open(image_path)
        positions = self.files[idx]
        positions = list(map(lambda x: int(x), re.findall(r'\d+', positions)))
        if self.transform:
            im = self.transform(im)
        return im, np.array(positions)