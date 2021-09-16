from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image


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

class OnlineDataset(Dataset):
    def __init__(self, data_path_name, transform=None):
        self.path = data_path_name
        self.files = os.listdir(data_path_name)
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.path, self.files[idx])
        im = Image.open(image_path)
        im_list, position_list = onlinecutpatches(im)
        if self.transform:
            for patch_id in range(len(im_list)):
                im_list[patch_id] = self.transform(im_list[patch_id])
        # label = int(self.files[idx][-5:-4])
        # position = tuple(int(self.files[idx][-7]), int(self.files[idx][-8]))
        return image_path, im_list, position_list

# Todo: define the onlinecutpatches function
def onlinecutpatches(im):
    return