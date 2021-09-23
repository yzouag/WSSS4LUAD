from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image
import torch


class SingleLabelDataset(Dataset):
    def __init__(self, data_path_name, transform=None):
        self.path = data_path_name
        self.files = os.listdir(data_path_name)
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.path, self.files[idx])
        im = Image.open(image_path)
        # im = im / 255 # convert to 0-1 scale
        if self.transform:
            im = self.transform(im)
        label = int(self.files[idx][-5:-4])
        return im, label

class DoubleLabelDataset(Dataset):
    def __init__(self, transform=None):
        self.path_s = "sample_single_patches"
        self.path_d = "train_double_patches"
        self.files_s = os.listdir(self.path_s)
        self.files_d = os.listdir(self.path_d)
        self.transform = transform

    def __len__(self):
        return len(self.files_s) + len(self.files_d)

    def __getitem__(self, idx):
        if idx < 15000:
            image_path = os.path.join(self.path_s, self.files_s[idx])
            im = Image.open(image_path)
            if self.transform:
                im = self.transform(im)
            label = np.array([0,0,0])
            activate = int(self.files_s[idx][-5])
            label[activate] = 1
        else:
            idx -= 15000
            image_path = os.path.join(self.path_d, self.files_d[idx])
            im = Image.open(image_path)
            if self.transform:
                im = self.transform(im)
            label = np.array([int(self.files_d[idx][-10]), int(self.files_d[idx][-8]), int(self.files_d[idx][-6])])
        return im, torch.tensor(label, requires_grad=False)

class DoubleValidDataset(Dataset):
    def __init__(self, transform=None):
        self.path_d = "valid_double_patches"
        self.files_d = os.listdir(self.path_d)
        self.transform = transform

    def __len__(self):
        return len(self.files_d)

    def __getitem__(self, idx):
        image_path = os.path.join(self.path_d, self.files_d[idx])
        im = Image.open(image_path)
        if self.transform:
            im = self.transform(im)
        label = np.array([int(self.files_d[idx][-10]), int(self.files_d[idx][-8]), int(self.files_d[idx][-6])])
        return im, torch.tensor(label, requires_grad=False)

class OriginPatchesDataset(Dataset):
    def __init__(self, data_path_name = "Dataset/1.training", transform=None):
        self.path = data_path_name
        # Need to eliminate the validation part from training
        self.files = os.listdir(data_path_name)
        # sample_index = np.load("sample_index.npy")
        # self.files = np.delete(origintrain, sample_index)
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

class OriginVaidationDataset(Dataset):
    def __init__(self, transform=None):
        self.path_v = "Dataset/2.validation"
        # self.path_t = "Dataset/1.training"
        self.files_v = os.listdir(os.path.join("Dataset/2.validation", "img"))[:30]
        # Use the pre-generated random number
        # sample_index = np.load("sample_index.npy")
        # self.files_t = np.array(os.listdir("Dataset/1.training"))[sample_index]
        self.files_b = os.listdir(os.path.join("Dataset/2.validation", "bigimg"))
        
        self.transform = transform

    def __len__(self):
        return len(self.files_v)  + len(self.files_b)

    def __getitem__(self, idx):
        if idx < 30:
            image_path = os.path.join(self.path_v, "img", self.files_v[idx])
            label_path = os.path.join(self.path_v, "mask", self.files_v[idx])
            im = Image.open(image_path)
            label_arr = np.asarray(Image.open(label_path))

            if self.transform:
                im = self.transform(im)
            s = set()
            for i in range(label_arr.shape[0]):
                for j in range(label_arr.shape[1]):
                    s.add(label_arr[i][j])

            label = np.array([1 if 0 in s else 0, 1 if 1 in s else 0, 1 if 2 in s else 0])
        # elif idx < 200:
        #     # for the data from training set
        #     idx = idx - 30
        #     image_path = os.path.join(self.path_t, self.files_t[idx])
        #     im = Image.open(image_path)

        #     if self.transform:
        #         im = self.transform(im)
        #     label = np.array([int(self.files_t[idx][-12]), int(self.files_t[idx][-9]), int(self.files_t[idx][-6])])
        else: # here we need big validation images
            idx = idx - 30
            image_path = os.path.join(self.path_v, "bigimg", self.files_b[idx])
            label_path = os.path.join(self.path_v, "bigmask", self.files_b[idx])
            im = Image.open(image_path)
            label_arr = np.asarray(Image.open(label_path))

            if self.transform:
                im = self.transform(im)
            # s = set()
            # for i in range(label_arr.shape[0]):
            #     for j in range(label_arr.shape[1]):
            #         s.add(label_arr[i][j])
            bin = np.bincount(label_arr.flatten())
            assert len(bin) == 4

            # label = np.array([1 if 0 in s else 0, 1 if 1 in s else 0, 1 if 2 in s else 0])
            label = np.array([1 if bin[0] > 6000 else 0, 1 if bin[1] > 6000 else 0, 1 if bin[2] > 6000 else 0])
        return im, label

class OriginVaidationNoMixDataset(Dataset):
    def __init__(self, transform=None):
        self.path_v = "Dataset/2.validation"
        self.files_v = os.listdir(os.path.join("Dataset/2.validation", "img"))[:30]
        self.transform = transform

    def __len__(self):
        return len(self.files_v)

    def __getitem__(self, idx):
        image_path = os.path.join(self.path_v, "img", self.files_v[idx])
        label_path = os.path.join(self.path_v, "mask", self.files_v[idx])
        im = Image.open(image_path)
        label_arr = np.asarray(Image.open(label_path))

        if self.transform:
            im = self.transform(im)
        s = set()
        for i in range(label_arr.shape[0]):
            for j in range(label_arr.shape[1]):
                s.add(label_arr[i][j])

        label = np.array([1 if 0 in s else 0, 1 if 1 in s else 0, 1 if 2 in s else 0])
        # print(str(idx), label)

        return im, label

class OnlineDataset(Dataset):
    def __init__(self, data_path_name, transform=None, patch_size = 56, stride=28):
        self.path = data_path_name
        self.files = os.listdir(data_path_name)
        self.transform = transform
        self.patch_size = patch_size
        self.stride = stride

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.path, self.files[idx])
        im = np.asarray(Image.open(image_path))
        im_list, position_list = online_cut_patches(im, self.patch_size, self.stride)
        if self.transform:
            for patch_id in range(len(im_list)):
                im_list[patch_id] = self.transform(im_list[patch_id])
        # label = int(self.files[idx][-5:-4])
        # position = tuple(int(self.files[idx][-7]), int(self.files[idx][-8]))
        return image_path, im_list, position_list

class OnlineTrainDataset(Dataset):
    def __init__(self, data_path_name, transform=None):
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

        return image_path, im 

def online_cut_patches(im, im_size=56, stride=28):
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

    h_ = np.arange(0, h - im_size + 1, stride)
    if h % stride != 0:
        h_ = np.append(h_, h-im_size)
    w_ = np.arange(0, w - im_size + 1, stride)
    if w % stride != 0:
        w_ = np.append(w_, w - im_size)

    for i in h_:
        for j in w_:   	
            temp = Image.fromarray(np.uint8(im[i:i+im_size,j:j+im_size,:].copy()))
            im_list.append(temp)
            position_list.append((i,j))
    return im_list, position_list
