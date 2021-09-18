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
parser.add_argument("--batch", default=64, type=int)
parser.add_argument('-d','--device', nargs='+', help='GPU id to use parallel', required=True, type=int)
parser.add_argument('-t', type=float, default = 0.8, required=False, help='the threshold probability to set the label of the image to 1')
args = parser.parse_args()

batch_size = args.batch
devices = args.device
threshold = args.t
base_lr = 0.0003
net = network.ResNet().cuda()

# Get pretrained model
resnet101 = models.resnet101(pretrained=True) 
pretrained_dict =resnet101.state_dict()
model_dict = net.state_dict()
pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k in model_dict} 
model_dict.update(pretrained_dict)
# Load pretraiend parameters
net.load_state_dict(model_dict)

net = torch.nn.DataParallel(net, device_ids=devices).cuda()
net.train()

TrainDataset = dataset.OriginPatchesDataset(transform=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))

print("Dataset", len(TrainDataset))
TrainDatasampler = torch.utils.data.RandomSampler(TrainDataset)
TrainDataloader = DataLoader(TrainDataset, batch_size=batch_size, num_workers=2, sampler=TrainDatasampler, drop_last=True)
optimizer = torch.optim.Adam(net.parameters(), base_lr, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.7)
criteria = torch.nn.BCEWithLogitsLoss(reduction='mean')

criteria.cuda()

epochs = 40
loss_g = []
accuracy_g = []

for i in range(epochs):
    running_loss = 0.
    count = 0
    correct = 0
    helpdic = {"tumor":[0, 0, 0], "stroma":[0, 0, 0], "normal":[0, 0, 0]} # TP, FP, FN

    for img, label in tqdm(TrainDataloader):
        count += 1
        img = img.cuda()
        
        scores = net(img) # probability of n * 3
        onehot_label = label.cuda() # gt label of n * 3
        loss = criteria(scores, onehot_label.float())
        
        predict = scores>=threshold # check dtype here
        for k in range(len(onehot_label)):
            if torch.equal(onehot_label[i], predict[i]):
                correct += 1
        
        # Calculate for the three statistics
        for index, tissue in enumerate(["tumor", "stroma", "normal"]):
            predict_type = predict[:, index].bool()
            gt_type = onehot_label[:, index].bool()
            helpdic[tissue][0] += gt_type[predict_type].sum().item()
            helpdic[tissue][1] += (~gt_type[predict_type]).sum().item()
            helpdic[tissue][2] += (~predict_type[gt_type]).sum().item()
        
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        running_loss += loss.item()

    scheduler.step()
    print("loss: ", running_loss / count)
    print("accuracy: ", correct / (count * batch_size))
    for tissue in ["tumor", "stroma", "normal"]:
        print("precision for", tissue, helpdic[tissue][0] / (helpdic[tissue][0] + helpdic[tissue][1]))
        print("recall for", tissue, helpdic[tissue][0] / (helpdic[tissue][0] + helpdic[tissue][2]))
    accuracy_g.append(correct / (count * batch_size))
    loss_g.append(running_loss / count)
    if (i + 1) % 10 == 0 and (i + 1) != epochs:
        torch.save({"model": net.state_dict(), 'optimizer': optimizer.state_dict()}, "./modelstates/bigpatch_model_ep"+str(i+1)+".pth")

fig=plt.figure()
plt.plot(loss_g)
plt.ylabel('loss')
plt.xlabel('epochs')
plt.savefig('./image/bigpatch_loss.png')
torch.save({"model": net.state_dict(), 'optimizer': optimizer.state_dict()}, "./modelstates/bigpatch_model_last.pth")

fig=plt.figure()
plt.plot(accuracy_g)
plt.ylabel('accuracy')
plt.xlabel('epochs')
plt.savefig('./image/bigpatch_accuracy.png')