import os
# os.environ['CUDA_VISIBLE_DEVICES']='2, 3, 4, 5, 6, 7'
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

def convertinttoonehot(nums_list):
    dic = {0: [1, 0, 0], 1: [0, 1, 0], 2: [0, 0, 1]}
    result = np.empty((len(nums_list), 3))
    for i in range(len(nums_list)):
        result[i] = np.array(dic[nums_list[i].item()])

    return torch.tensor(result, requires_grad=False)


parser = argparse.ArgumentParser()
parser.add_argument("--batch", default=50, type=int)
parser.add_argument('-d','--device', nargs='+', help='GPU id to use parallel', required=True)
parser.add_argument('-setting', type=str, help='the stride and pathsize setting', required=True)
parser.add_argument("-ce", action='store_true', help='whether to use ce loss')
args = parser.parse_args()

batch_size = args.batch
devices = args.d
setting_str = args.setting
use_ce = args.ce
base_lr = 0.0003
net = network.ResNet().cuda()
assert os.path.exists("./train_single_patches"), "The directory train_single_patches haven't been genereated!"

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

TrainDataset = dataset.SingleLabelDataset("train_single_patches/", transform=transforms.Compose([
    transforms.Resize(224),
    # transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))

print("Dataset", len(TrainDataset))
TrainDatasampler = torch.utils.data.RandomSampler(TrainDataset)
TrainDataloader = DataLoader(TrainDataset, batch_size=batch_size, num_workers=2, sampler=TrainDatasampler, drop_last=True)
optimizer = torch.optim.Adam(net.parameters(), base_lr, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)
if use_ce:
    criteria = torch.nn.CrossEntropyLoss(reduction='mean')
else:
    criteria = torch.nn.BCEWithLogitsLoss(reduction='mean')

criteria.cuda()

epochs = 15
loss_g = []
accuracy_g = []

for i in range(epochs):
    running_loss = 0.
    count = 0
    correct = 0

    for img, label in tqdm(TrainDataloader):
        count += 1
        img = img.cuda()
        
        scores = net(img)
        if use_ce:
            label = label.cuda()
            loss = criteria(scores, label)
        else:
            onehot_label = convertinttoonehot(label).cuda()
            loss = criteria(scores, onehot_label)
            label = label.cuda()
        
        level = scores.detach().argmax(dim = 1)
        b = level == label
        correct += torch.sum(b).item()

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        running_loss += loss.item()

    scheduler.step()
    print("loss: ", running_loss / count)
    print("accuracy: ", correct / (count * batch_size))
    accuracy_g.append(correct / (count * batch_size))
    loss_g.append(running_loss / count)
    if (i + 1) % 5 == 0 and (i + 1) != epochs:
        torch.save({"model": net.state_dict(), 'optimizer': optimizer.state_dict()}, "./modelstates/ce_" + setting_str + "_model_ep"+str(i+1)+".pth")

fig=plt.figure()
plt.plot(loss_g)
plt.ylabel('loss')
plt.xlabel('epochs')
plt.savefig('./image/ce_' + setting_str + '_loss.png')
torch.save({"model": net.state_dict(), 'optimizer': optimizer.state_dict()}, "./modelstates/ce_" + setting_str + "_model_last.pth")

fig=plt.figure()
plt.plot(accuracy_g)
plt.ylabel('accuracy')
plt.xlabel('epochs')
plt.savefig('./image/ce_' + setting_str + '_accuracy.png')