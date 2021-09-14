from torch.utils.data import Dataset
import os
from cv2 import imread


class SingleLabelDataset(Dataset):
    def __init__(self, data_path_name):
        self.path = data_path_name
        self.files = os.listdir(data_path_name)


    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.path, self.files[idx])
        im = imread(image_path)
        label = self.files[idx][-5:-4]
        return im, label

# class RandomCrop(object):
#     """
#     Crop randomly the image in a sample
#     Args:
#     output_size (int): Desired output size
#     """

#     def __init__(self, output_size):
#         self.output_size = output_size  # (c, w, h, d)

#     def __call__(self, sample):
#         img = sample["image"]
#         (c, w, h, d) = img.shape

#         w1 = np.random.randint(0, w - self.output_size[1] + 1)
#         h1 = np.random.randint(0, h - self.output_size[2] + 1)
#         d1 = np.random.randint(0, d - self.output_size[3] + 1)

#         img = img[:, w1:w1 + self.output_size[1], h1:h1 + self.output_size[2], d1:d1 + self.output_size[3]]

#         return {'image': img, 'label': sample["label"]}

# class CenterCrop(object):
#     """
#     Crop center of the image in a sample
#     Args:
#     output_size (int): Desired output size
#     """

#     def __init__(self, output_size):
#         self.output_size = output_size  # (c, w, h, d)

#     def __call__(self, sample):
#         img = sample["image"]
#         (c, w, h, d) = img.shape

#         w1 = (w - self.output_size[1]) // 2
#         h1 = (h - self.output_size[2]) // 2
#         d1 = (d - self.output_size[3]) // 2

#         img = img[:, w1:w1 + self.output_size[1], h1:h1 + self.output_size[2], d1:d1 + self.output_size[3]]

#         return {'image': img, 'label': sample["label"]}

# class RandomFlip(object):
#     """
#     flip the image in given dimension
#     """

#     def __call__(self, sample):
#         img, label = sample["image"], sample['label']
#         if np.random.rand() < 0.5:
#             axis = np.random.randint(0, 2)
#             axis += 2
#             img = np.flip(img, axis=axis).copy()

#         return {'image': img, 'label': label}

# class RandomRot(object):
#     """
#     Rotate the image in given dimension
#     """

#     def __call__(self, sample):
#         img, label = sample["image"], sample['label']
#         if np.random.rand() < 0.5:
#             img = np.rot90(img, 1, axes=(2, 3))

#         return {'image': img, 'label': label}

# class ToTensor(object):
#     """Convert ndarrays in sample to Tensors."""

#     def __call__(self, sample):
#         image = sample['image']
#         img = image[0:3:2].astype(np.float32)
#         segmentation = image[1::2].astype(np.float32)
#         # print("segment shape", segmentation.shape)

#         return {'image': torch.from_numpy(img), 'segment':torch.from_numpy(segmentation), 'label': sample['label']}