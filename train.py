import torch
import network
import dataset
import os
from torch.utils.data import DataLoader
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm
import torchvision.models as models
# import train_loss

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
batch_size = 50
net = network.ResNet().cuda()

# Get pretrained model
resnet101 = models.resnet101(pretrained=True) 
pretrained_dict =resnet101.state_dict()
model_dict = net.state_dict()
# print("Pretrained:", len(pretrained_dict.keys()))
# print("Model:", len(model_dict.keys()))
not_exist = [k for k in model_dict.keys() if not k in pretrained_dict.keys()]
print("not exists:", not_exist)
print("weights:", model_dict[not_exist[-2]])
pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k in model_dict} 
# print(len(pretrained_dict))
model_dict.update(pretrained_dict)
# Load pretraiend parameters
net.load_state_dict(model_dict)

net.cuda()
net.train()

Dataset = dataset.SingleLabelDataset("train_single_patches/", transform=transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))

print("Dataset", len(Dataset))
Dataloader = DataLoader(Dataset, batch_size=batch_size, num_workers=2)

for img, label in Dataloader:

    img = img.cuda()
    print("img shape is:", img.shape)

    scores = net(img)
    print(scores.shape)
    break

print('Testing performance is: accuracy : %f' % (3))