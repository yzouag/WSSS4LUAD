from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image
from utils.util import multiscale_online_crop
class OriginPatchesDataset(Dataset):
    def __init__(self, data_path_name = "Dataset/1.training", transform=None):
        self.path = data_path_name
        self.files = os.listdir(data_path_name)
        self.transform = transform

    def __len__(self):
        # return len(self.files)
        return 50

    def __getitem__(self, idx):
        image_path = os.path.join(self.path, self.files[idx])
        im = Image.open(image_path)

        if self.transform:
            im = self.transform(im)
        label = np.array([int(self.files[idx][-12]), int(self.files[idx][-9]), int(self.files[idx][-6])])
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
    def __init__(self, data_path_name, transform=None, patch_size = 224, stride=74, scales=[0.5, 0.75, 1, 1.25, 1.5]):
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
