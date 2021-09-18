import os
os.environ['CUDA_VISIBLE_DEVICES']='2'
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
parser.add_argument("--batch", default=32, type=int)
parser.add_argument("-epoch", default=100, type=int)
# parser.add_argument('-d','--device', nargs='+', help='GPU id to use parallel', required=True, type=int)
parser.add_argument('-t', type=float, default = 0.9, required=False, help='the threshold probability to set the label of the image to 1')
# parser.add_argument("-v", action='store_true', help='whether it is to validate')
args = parser.parse_args()

batch_size = args.batch
# devices = args.device
threshold = args.t
epochs = args.epoch
base_lr = 0.0001
net = network.ResNet().cuda()

# Get pretrained model
resnet101 = models.resnet101(pretrained=True) 
pretrained_dict =resnet101.state_dict()
model_dict = net.state_dict()
pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k in model_dict} 
model_dict.update(pretrained_dict)
# Load pretraiend parameters
net.load_state_dict(model_dict)

# net = torch.nn.DataParallel(net, device_ids=devices).cuda()
net = net.cuda()

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
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.955)
criteria = torch.nn.BCEWithLogitsLoss(reduction='mean')

criteria.cuda()

loss_g = []
accuracy_g = []
valid_accuracy = []

for i in range(epochs):
    running_loss = 0.
    count = 0
    correct = 0
    helpdic = {"tumor":[0, 0, 0], "stroma":[0, 0, 0], "normal":[0, 0, 0]} # TP, FP, FN
    net.train()

    for img, label in tqdm(TrainDataloader):
        count += 1
        img = img.cuda()
        
        scores = net(img) # probability of n * 3
        onehot_label = label.cuda() # gt label of n * 3
        loss = criteria(scores, onehot_label.float())
        
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        running_loss += loss.item()
        predict = scores.detach()>=threshold # check dtype here
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

    validDataset = dataset.OriginVaidationDataset(transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))

    print("Dataset", len(validDataset))
    ValidDataloader = DataLoader(validDataset, batch_size=20, num_workers=2, drop_last=False)
    net.eval()
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

    print("validation accuracy: ", correct / count)
    valid_accuracy.append(correct / count)
    for tissue in ["tumor", "stroma", "normal"]:
        print("validation precision for", tissue, helpdic[tissue][0] / (helpdic[tissue][0] + helpdic[tissue][1]))
        print("validation recall for", tissue, helpdic[tissue][0] / (helpdic[tissue][0] + helpdic[tissue][2]))


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

fig=plt.figure()
plt.plot(valid_accuracy)
plt.ylabel('validation accuracy')
plt.xlabel('epochs')
plt.savefig('./image/bigpatch_validaccuracy.png')