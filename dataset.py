from PIL import Image
from torch.utils.data import Dataset
import os
import numpy as np
import torch
import torch.nn.functional as F


class SingleLabelDataset(Dataset):

    def __init__(self, folders, transform=None):
        self.folders, self.labels = self.get_labels(folders)
        self.transform = transform

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, idx):
        folder_id = self.folders[idx]

        with open(folder_id, 'rb') as f:
            color_img = pickle.load(f)


        # img process
        c1 = color_img[:, :, 0]
        c2 = color_img[:, :, 1]
        c3 = color_img[:, :, 2]
        c4 = color_img[:, :, 3]

        c1 = np.expand_dims(c1, axis=2)
        c2 = np.expand_dims(c2, axis=2)
        c3 = np.expand_dims(c3, axis=2)
        c4 = np.expand_dims(c4, axis=2)

        color_img = np.concatenate((c1, c2, c3, c4), axis=2)

        # img process, transform
        # color_img = np.uint8(255 * color_img)
        # color_img = Image.fromarray(color_img)
        if self.transform is not None:
            color_img = np.uint8(255 * color_img)
            color_img = Image.fromarray(color_img)
            color_img = self.transform(color_img)
        return color_img, self.labels[idx]

    def get_labels(self, folders):

        files = []
        labels = []

        # conding=utf8
        g = os.walk(folders)

        for path, _, file_list in g:
            for file_name in file_list:
                files.append(os.path.join(path, file_name))
                if 'nil_HS_H08' in file_name or 'light_HS_H08' in file_name:
                    labels.append(0)
                elif 'moderate_HS_H08' in file_name:
                    labels.append(1)
                else:
                    labels.append(2)

        return files, labels


class RandomCrop(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size):
        self.output_size = output_size  # (c, w, h, d)

    def __call__(self, sample):
        img = sample["image"]
        (c, w, h, d) = img.shape

        w1 = np.random.randint(0, w - self.output_size[1] + 1)
        h1 = np.random.randint(0, h - self.output_size[2] + 1)
        d1 = np.random.randint(0, d - self.output_size[3] + 1)

        img = img[:, w1:w1 + self.output_size[1], h1:h1 + self.output_size[2], d1:d1 + self.output_size[3]]

        return {'image': img, 'label': sample["label"]}

class CenterCrop(object):
    """
    Crop center of the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size):
        self.output_size = output_size  # (c, w, h, d)

    def __call__(self, sample):
        img = sample["image"]
        (c, w, h, d) = img.shape

        w1 = (w - self.output_size[1]) // 2
        h1 = (h - self.output_size[2]) // 2
        d1 = (d - self.output_size[3]) // 2

        img = img[:, w1:w1 + self.output_size[1], h1:h1 + self.output_size[2], d1:d1 + self.output_size[3]]

        return {'image': img, 'label': sample["label"]}

class RandomFlip(object):
    """
    flip the image in given dimension
    """

    def __call__(self, sample):
        img, label = sample["image"], sample['label']
        if np.random.rand() < 0.5:
            axis = np.random.randint(0, 2)
            axis += 2
            img = np.flip(img, axis=axis).copy()

        return {'image': img, 'label': label}

class RandomRot(object):
    """
    Rotate the image in given dimension
    """

    def __call__(self, sample):
        img, label = sample["image"], sample['label']
        if np.random.rand() < 0.5:
            img = np.rot90(img, 1, axes=(2, 3))

        return {'image': img, 'label': label}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample['image']
        img = image[0:3:2].astype(np.float32)
        segmentation = image[1::2].astype(np.float32)
        # print("segment shape", segmentation.shape)

        return {'image': torch.from_numpy(img), 'segment':torch.from_numpy(segmentation), 'label': sample['label']}