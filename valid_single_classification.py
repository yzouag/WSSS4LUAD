import os
os.environ['CUDA_VISIBLE_DEVICES']='3'
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
# net = network.ResNet()
net = network.scalenet152(structure_path='structures/scalenet152.json')

for m in ["scalenet152_ep5", "scalenet152_ep10", "scalenet152_last"]:

    path = "./modelstates/" + m + ".pth"
    pretrained = torch.load(path)['model']
    pretrained = {k[7:] : v for k, v in pretrained.items()}
    net.load_state_dict(pretrained)
    print(f'Model loaded from {path}')
    net.cuda()
    net.eval()

    validDataset = dataset.DoubleValidDataset(transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))

    print("Dataset", len(validDataset))
    ValidDataloader = DataLoader(validDataset, batch_size=1, num_workers=2, drop_last=True)
    criteria = torch.nn.BCEWithLogitsLoss(reduction='mean').cuda()
    correct = 0
    count = 0
    loss_t = 0
    with torch.no_grad():

        for inputs, labels in tqdm(ValidDataloader):
            labels = labels.cuda()
            if torch.sum(labels) > 1:
                continue
            inputs = inputs.cuda()
            scores = net(inputs)
            # print("scores", scores)
            # print("label", labels)
            loss = criteria(scores, labels.float())
            loss_t += loss.item()

            predict = torch.argmax(scores)
            if predict == torch.argmax(labels):
                # print(predict)
                correct += 1
            count += 1

    print("accuracy for validation is: ", (correct / count))
    print("The loss is:", loss_t / (count / 1))
