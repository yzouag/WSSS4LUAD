import os
from threading import current_thread
# os.environ['CUDA_VISIBLE_DEVICES']='2'
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
parser.add_argument("-epoch", default=40, type=int)
parser.add_argument('-d','--device', nargs='+', help='GPU id to use parallel', required=True, type=int)
parser.add_argument('-t', type=float, default = 0.9, required=False, help='the threshold probability to set the label of the image to 1')
args = parser.parse_args()

batch_size = args.batch
devices = args.device
threshold = args.t
epochs = args.epoch
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
# net = net.cuda()

TrainDataset = dataset.OriginPatchesDataset(transform=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))

print("Dataset", len(TrainDataset))

validDataset = dataset.OriginVaidationDataset(transform=transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))

TrainDatasampler = torch.utils.data.RandomSampler(TrainDataset)
TrainDataloader = DataLoader(TrainDataset, batch_size=batch_size, num_workers=2, sampler=TrainDatasampler, drop_last=True)
optimizer = torch.optim.Adam(net.parameters(), base_lr, weight_decay=2e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.95)
criteria = torch.nn.BCEWithLogitsLoss(reduction='mean')

criteria.cuda()

loss_g = []
accuracy_g = []
valid_accuracy = []
threshold_list = []
f1_scores = []
best_threshold = []
best_f1mean = 0
# follow_accuracy = 0

for i in range(epochs):
    print('This is epoch', i)
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
        scores = torch.sigmoid(scores)
        predict = scores.detach() >= threshold # check dtype here
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
        torch.save({"model": net.state_dict(), 'optimizer': optimizer.state_dict()}, "./modelstates/bigpatch6000_model_ep"+str(i+1)+".pth")

    print("Dataset", len(validDataset))
    ValidDataloader = DataLoader(validDataset, batch_size=24, num_workers=2, drop_last=False)
    net.eval()

    with torch.no_grad():
        helpdic = {"tumor":[0, 0, 0, 0, 0], "stroma":[0, 0, 0, 0, 0], "normal":[0, 0, 0, 0, 0]} # TP, FP, FN, bestf1, threshold
        remember_all_predict = []
        remember_all_label = []

        for img, label in tqdm(ValidDataloader):
            # count += img.shape[0]
            img = img.cuda()
            
            scores = net(img) # probability of n * 3
            onehot_label = label.cuda() # gt label of n * 3
            scores = torch.sigmoid(scores)
            remember_all_predict.append(scores.detach())
            remember_all_label.append(onehot_label.detach())

    # iter through validdataloader, start grid search

    # stack the list first
    remember_all_predict = torch.cat(remember_all_predict, dim=0)
    remember_all_label = torch.cat(remember_all_label, dim=0)
    count = len(remember_all_label)

    for threshold in tqdm(np.arange(0, 1, step = 0.01)):
        predict = remember_all_predict >= threshold

        # Calculate for the three statistics
        for index, tissue in enumerate(["tumor", "stroma", "normal"]):
            predict_type = predict[:, index].bool()
            gt_type = remember_all_label[:, index].bool()
            helpdic[tissue][0] = gt_type[predict_type].sum().item()
            helpdic[tissue][1] = (~gt_type[predict_type]).sum().item()
            helpdic[tissue][2] = (~predict_type[gt_type]).sum().item()
            precision = helpdic[tissue][0] / (helpdic[tissue][0] + helpdic[tissue][1] + 1e-6)
            recall = helpdic[tissue][0] / (helpdic[tissue][0] + helpdic[tissue][2] + 1e-6)
            f1_score = 2 * precision * recall / (precision + recall + 1e-6)
            if f1_score > helpdic[tissue][3]:
                helpdic[tissue][3] = f1_score
                helpdic[tissue][4] = threshold
            

    current_f1mean =  (helpdic["tumor"][3] + helpdic["stroma"][3] + helpdic["normal"][3]) / 3
    current_t =  np.array([helpdic["tumor"][4], helpdic["stroma"][4], helpdic["normal"][4]])
    if current_f1mean > best_f1mean:
        print("updating............................................................................................")
        best_f1mean = current_f1mean
        best_threshold = np.array([helpdic["tumor"][4], helpdic["stroma"][4], helpdic["normal"][4]])
        # follow_accuracy = accuracy
        torch.save({"model": net.state_dict(), 'optimizer': optimizer.state_dict()}, "./modelstates/bigpatch6000_model_best.pth")
    
    correct = 0
    temp = remember_all_predict >= torch.from_numpy(current_t).cuda()
    for k in range(len(temp)):
        if torch.equal(remember_all_label[k], temp[k]):
            correct += 1
    accuracy = correct / count
    
    print("validation current accuracy: ", accuracy)
    print("validation best threshold: ", best_threshold)
    print("validation best f1 mean: ", best_f1mean)
    print("validation current f1 mean: ", current_f1mean)
    print("validation current threshold: ", current_t)
    valid_accuracy.append(accuracy)

    f1_scores.append(current_f1mean)

fig=plt.figure()
plt.plot(loss_g)
plt.ylabel('loss')
plt.xlabel('epochs')
plt.savefig('./image/bigpatch6000_loss.png')
torch.save({"model": net.state_dict(), 'optimizer': optimizer.state_dict()}, "./modelstates/bigpatch6000_model_last.pth")

fig=plt.figure()
plt.plot(accuracy_g)
plt.ylabel('accuracy')
plt.xlabel('epochs')
plt.savefig('./image/bigpatch6000_accuracy.png')

fig=plt.figure()
plt.plot(valid_accuracy)
plt.ylabel('validation accuracy')
plt.xlabel('epochs')
plt.savefig('./image/bigpatch6000_validaccuracy.png')

fig=plt.figure()
plt.plot(f1_scores)
plt.ylabel('validation mean f1 scores')
plt.xlabel('epochs')
plt.savefig('./image/bigpatch6000_f1mean.png')

# fig=plt.figure()
# plt.plot(threshold_list)
# plt.ylabel('validation threshold')
# plt.xlabel('epochs')
# plt.savefig('./image/bigpatch_threshold.png')

# threshold_list = np.array(threshold_list)
# np.save("./threshold.npy", threshold_list)