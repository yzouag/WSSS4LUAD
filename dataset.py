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

# class SingleLabelValidationDataset(Dataset):
#     def __init__(self, data_path_name, transform=None):
#         self.path = data_path_name
#         self.files = os.listdir(data_path_name)
#         self.transform = transform

#     def __len__(self):
#         return len(self.files)

#     def __getitem__(self, idx):
#         image_path = os.path.join(self.path, self.files[idx])
#         im = Image.open(image_path)
#         # im = im / 255 # convert to 0-1 scale
#         if self.transform:
#             im = self.transform(im)
#         label = int(self.files[idx][-5:-4])
#         return im, label
# we don't need self-designed transform functions

# class RandomFlip(object):
#     """
#     flip the image in given dimension
#     """

#     def __call__(self, im):
#         if np.random.rand() < 0.5:
#             axis = np.random.randint(0, 2)
#             axis += 1
#             im = np.flip(im, axis=axis).copy()

#         return im

# class RandomRot(object):
#     """
#     Rotate the image in given dimension
#     """

#     def __call__(self, im):
#         if np.random.rand() < 0.5:
#             im = np.rot90(im, 1, axes=(2, 3))

#         return im