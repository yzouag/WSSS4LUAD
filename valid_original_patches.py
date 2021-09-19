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
path = "./modelstates/bigpatch_model_best.pth"
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
    helpdic = {"tumor":[0, 0, 0], "stroma":[0, 0, 0], "normal":[0, 0, 0]} # TP, FP, FN
    remember_all_predict = []
    remember_all_label = []

    for img, label in tqdm(ValidDataloader):
        # count += img.shape[0]
        img = img.cuda()
        
        scores = net(img) # probability of n * 3
        onehot_label = label.cuda() # gt label of n * 3
        scores = F.sigmoid(scores)
        # assert torch.all(scores>=0)
        # assert torch.all(scores<=1)
        remember_all_predict.append(scores.detach())
        remember_all_label.append(onehot_label.detach())

    # iter through validdataloader, start grid search

    # stack the list first
    remember_all_predict = torch.cat(remember_all_predict, dim=0)
    remember_all_label = torch.cat(remember_all_label, dim=0)
    count = len(remember_all_label)
    assert count == 200, "error: tensor size not equal to dataset size!"

    best_threshold = 0
    best_f1mean = 0
    follow_accuracy = 0

    # for threshold in tqdm(np.arange(0, 1, step = 0.01)):
        # calculate accuracy
    threshold = torch.tensor([0.06, 0.07, 0.98]).cuda()
    correct = 0
    predict = remember_all_predict >= threshold
    for k in range(len(predict)):
        if torch.equal(remember_all_label[k], predict[k]):
            correct += 1
    accuracy = correct / count

        # Calculate for the three statistics
    temp_f1 = []
    for index, tissue in enumerate(["tumor", "stroma", "normal"]):
        predict_type = predict[:, index].bool()
        gt_type = remember_all_label[:, index].bool()
        helpdic[tissue][0] = gt_type[predict_type].sum().item()
        helpdic[tissue][1] = (~gt_type[predict_type]).sum().item()
        helpdic[tissue][2] = (~predict_type[gt_type]).sum().item()
        precision = helpdic[tissue][0] / (helpdic[tissue][0] + helpdic[tissue][1])
        recall = helpdic[tissue][0] / (helpdic[tissue][0] + helpdic[tissue][2])
        f1_score = 2 * precision * recall / (precision + recall)
        temp_f1.append(f1_score)
    
    current_f1mean = sum(temp_f1) / len(temp_f1)
    # if current_f1mean > best_f1mean:
    #     best_f1mean = current_f1mean
    #     best_threshold = threshold
    #     follow_accuracy = accuracy
            
    print("validation accuracy: ", accuracy)
    # print("validation threshold: ", best_threshold)
    # print("validation f1 mean: ", best_f1mean)
    print("validation f1 mean: ", current_f1mean)

    for tissue in ["tumor", "stroma", "normal"]:
        print("precision for", tissue, helpdic[tissue][0] / (helpdic[tissue][0] + helpdic[tissue][1]))
        print("recall for", tissue, helpdic[tissue][0] / (helpdic[tissue][0] + helpdic[tissue][2]))
        # print("fi_score for", tissue, helpdic[tissue][0] / (helpdic[tissue][0] + helpdic[tissue][2]))

