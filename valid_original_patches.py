import os
import torch
import network
import dataset
from torch.utils.data import DataLoader
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm, trange
import torchvision.models as models
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--batch", default=20, type=int)
parser.add_argument('-t', type=float, default = 0.8, required=False, help='the threshold probability to set the label of the image to 1')
args = parser.parse_args()

batch_size = args.batch
threshold = args.t
net = network.ResNet()
path = "./modelstates/bigpatch_model_last.pth"
pretrained = torch.load(path)['model']
pretrained_modify = {k[7:] : v for k, v in pretrained.items()}
net.load_state_dict(pretrained_modify)
print(f'Model loaded from {path}')
net.cuda()
net.eval()

validDataset = dataset.OriginVaidationDataset(transform=transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))

print("Dataset", len(validDataset))
ValidDataloader = DataLoader(validDataset, batch_size=batch_size, num_workers=2, drop_last=False)

with torch.no_grad():
    count = 0
    correct = 0
    helpdic = {"tumor":[0, 0, 0], "stroma":[0, 0, 0], "normal":[0, 0, 0]} # TP, FP, FN

    for img, label in tqdm(ValidDataloader):
        count += img.shape[0]
        img = img.cuda()
        
        scores = net(img) # probability of n * 3
        onehot_label = label.cuda() # gt label of n * 3
        
        predict = scores>=threshold # check dtype here
        for k in range(len(onehot_label)):
            if torch.equal(onehot_label[k], predict[k]):
                correct += 1
        
        # Calculate for the three statistics
        for index, tissue in enumerate(["tumor", "stroma", "normal"]):
            predict_type = predict[:, index].bool()
            gt_type = onehot_label[:, index].bool()
            helpdic[tissue][0] += gt_type[predict_type].sum().item()
            helpdic[tissue][1] += (~gt_type[predict_type]).sum().item()
            helpdic[tissue][2] += (~predict_type[gt_type]).sum().item()

    print("accuracy: ", correct / count)
    for tissue in ["tumor", "stroma", "normal"]:
        print("precision for", tissue, helpdic[tissue][0] / (helpdic[tissue][0] + helpdic[tissue][1]))
        print("recall for", tissue, helpdic[tissue][0] / (helpdic[tissue][0] + helpdic[tissue][2]))

