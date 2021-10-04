import os
os.environ['CUDA_VISIBLE_DEVICES']='2,3'
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
parser.add_argument("-epoch", default=30, type=int)
parser.add_argument("-lr", default=0.001, type=float)
parser.add_argument("-resize", default=448, type=int)
parser.add_argument("-step", default=10, type=int)
parser.add_argument("-save_every", default=10, type=int, help="how often to save a model while training")
parser.add_argument("-gamma", default=0.1, type=float)
parser.add_argument('-d','--device', nargs='+', help='GPU id to use parallel', required=True, type=int)
parser.add_argument('-m', type=str, help='the save model name', required=True)
args = parser.parse_args()

batch_size = args.batch
epochs = args.epoch
base_lr = args.lr
resize = args.resize
step_size = args.step
save_every = args.save_every
gamma = args.gamma
devices = args.device
setting_str = args.m

net = network.scalenet101(structure_path='../structures/scalenet101.json', ckpt='../weights/scalenet101.pth')
net = torch.nn.DataParallel(net, device_ids=devices).cuda()
net.train()

TrainDataset = dataset.DoubleLabelDataset(transform=transforms.Compose([
    transforms.Resize((resize, resize)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))

print("Dataset", len(TrainDataset))
TrainDatasampler = torch.utils.data.RandomSampler(TrainDataset)
TrainDataloader = DataLoader(TrainDataset, batch_size=batch_size, num_workers=2, sampler=TrainDatasampler, drop_last=True)
optimizer = torch.optim.SGD(net.parameters(), base_lr, momentum=0.9, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
criteria = torch.nn.BCEWithLogitsLoss(reduction='mean')
criteria.cuda()

loss_g = []
accuracy_g = []

for i in range(epochs):
    running_loss = 0.
    count = 0
    correct = 0

    for img, label in tqdm(TrainDataloader):
        count += 1
        img = img.cuda()
        label = label.cuda()
        scores = net(img)
        loss = criteria(scores, label.float())
        
        scores = torch.sigmoid(scores)
        predict = torch.zeros_like(scores)
        predict[scores > 0.7] = 1
        predict[scores < 0.3] = 0
        predict[torch.logical_and(scores > 0.3, scores < 0.7)] = -1
        for k in range(len(predict)):
            if torch.equal(predict[k], label[k]):
                correct += 1

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        running_loss += loss.item()

    scheduler.step()
    print("loss: ", running_loss / count)
    print("accuracy: ", correct / (count * batch_size))
    accuracy_g.append(correct / (count * batch_size))
    loss_g.append(running_loss / count)
    if (i + 1) % save_every == 0 and (i + 1) != epochs:
        torch.save({"model": net.state_dict(), 'optimizer': optimizer.state_dict()}, "./modelstates/" + setting_str + "_ep"+str(i+1)+".pth")

torch.save({"model": net.state_dict(), 'optimizer': optimizer.state_dict()}, "./modelstates/" + setting_str + "_last.pth")

fig=plt.figure()
plt.plot(loss_g)
plt.ylabel('loss')
plt.xlabel('epochs')
plt.savefig('./image/loss.png')

fig=plt.figure()
plt.plot(accuracy_g)
plt.ylabel('accuracy')
plt.xlabel('epochs')
plt.savefig('./image/accuracy.png')