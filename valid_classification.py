import os
os.environ['CUDA_VISIBLE_DEVICES']='0, 2'
import torch
import network
import dataset
from torch.utils.data import DataLoader
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm, trange
import torchvision.models as models
# import train_loss
import matplotlib.pyplot as plt
import argparse

# def convertinttoonehot(nums_list):
#     dic = {0: [1, 0, 0], 1: [0, 1, 0], 2: [0, 0, 1]}
#     result = np.empty((len(nums_list), 3))
#     for i in range(len(nums_list)):
#         result[i] = np.array(dic[nums_list[i].item()])

#     return torch.tensor(result, requires_grad=False)

parser = argparse.ArgumentParser()
parser.add_argument("--batch", default=50, type=int)
args = parser.parse_args()

batch_size = args.batch
net = network.ResNet()
path = "./model.pth"
pretrained = torch.load(path)
pretrained_modify = {k[7:] : v for k, v in pretrained.items()}
net.load_state_dict(pretrained_modify)
print(f'Model loaded from {path}')
net.cuda()
net.eval()

validDataset = dataset.SingleLabelDataset("valid_single_patches/", transform=transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))

print("Dataset", len(validDataset))
ValidDataloader = DataLoader(validDataset, batch_size=batch_size, num_workers=2, drop_last=True)

correct = 0
count = 0

with torch.no_grad():

    for inputs, labels in tqdm(ValidDataloader):
        labels = labels.cuda()
        inputs = inputs.cuda()
        scores = net(inputs)

        level = scores.detach().argmax(dim = 1)
        correct += torch.sum(level == labels).item()
        count += batch_size

print("accuracy for validation is: ", (correct / count))
