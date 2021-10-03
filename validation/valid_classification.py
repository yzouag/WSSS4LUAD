import os
os.environ['CUDA_VISIBLE_DEVICES']='2, 3'
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

parser = argparse.ArgumentParser()
parser.add_argument("--batch", default=32, type=int)
args = parser.parse_args()

batch_size = args.batch
net = network.ResNet()

for m in ["12856_ep10", "12856_ep20", "12856_last"]:

    path = "./modelstates/" + m + ".pth"
    pretrained = torch.load(path)['model']
    pretrained_modify = {k[7:] : v for k, v in pretrained.items()}
    net.load_state_dict(pretrained_modify)
    print(f'Model loaded from {path}')
    net.cuda()
    net.eval()

    validDataset = dataset.DoubleValidDataset(transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))

    print("Dataset", len(validDataset))
    ValidDataloader = DataLoader(validDataset, batch_size=batch_size, num_workers=2, drop_last=True)
    criteria = torch.nn.BCEWithLogitsLoss(reduction='mean').cuda()
    correct = 0
    count = 0
    loss_t = 0
    with torch.no_grad():

        for inputs, labels in tqdm(ValidDataloader):
            labels = labels.cuda()
            inputs = inputs.cuda()
            scores = net(inputs)
            # print("scores", scores)
            # print("label", labels)
            loss = criteria(scores, labels.float())
            loss_t += loss.item()

            scores = torch.sigmoid(scores)
            scores[scores >= 0.5] = 1
            scores[scores < 0.5] = 0
            # scores[torch.logical_and(scores > 0.3, scores < 0.7)] = -1
            for k in range(len(scores)):
                if torch.equal(scores[k], labels[k]):
                    correct += 1
            count += batch_size

    print("accuracy for validation is: ", (correct / count))
    print("The loss is:", loss_t / (count / batch_size))
