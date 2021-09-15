import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
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

def convertinttoonehot(nums_list):
    dic = {0: [1, 0, 0], 1: [0, 1, 0], 2: [0, 0, 1]}
    result = np.empty((len(nums_list), 3))
    for i in range(len(nums_list)):
        result[i] = np.array(dic[nums_list[i].item()])

    return torch.tensor(result, requires_grad=False)


parser = argparse.ArgumentParser()
parser.add_argument("--batch", default=50, type=int)
args = parser.parse_args()

batch_size = args.batch
base_lr = 0.001
net = network.ResNet().cuda()

# Get pretrained model
resnet101 = models.resnet101(pretrained=True) 
pretrained_dict =resnet101.state_dict()
model_dict = net.state_dict()
# print("Pretrained:", len(pretrained_dict.keys()))
# print("Model:", len(model_dict.keys()))
# not_exist = [k for k in model_dict.keys() if not k in pretrained_dict.keys()]
# print("not exists:", not_exist)
# print("weights:", model_dict[not_exist[-2]])
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
Datasampler = torch.utils.data.RandomSampler(Dataset)
Dataloader = DataLoader(Dataset, batch_size=batch_size, num_workers=2, sampler=Datasampler, drop_last=True)
optimizer = torch.optim.Adam(net.parameters(), base_lr, weight_decay=1e-4)
criteria = torch.nn.BCEWithLogitsLoss(reduction='mean')
criteria.cuda()

epochs = 30
loss_g = []
accuracy_g = []

for i in trange(epochs):
    running_loss = 0.
    count = 0
    correct = 0

    for img, label in Dataloader:
        count += 1
        img = img.cuda()
        onehot_label = convertinttoonehot(label).cuda()

        scores = net(img)
        
        loss = criteria(scores, onehot_label)
        level = scores.detach().argmax(dim = 1)
        label = label.cuda()
        b = level == label
        correct += torch.sum(b).item()
        print("level: ", torch.sum(b).item())
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        running_loss += loss.item()

    print("loss: ", running_loss / count)
    print("accuracy: ", correct / (count * batch_size))
    accuracy_g.append(correct / (count * batch_size))
    loss_g.append(running_loss / count)

fig=plt.figure()
plt.plot(loss_g)
plt.ylabel('loss')
plt.xlabel('epochs')
plt.savefig('loss.png')
torch.save(net.state_dict(), "./model.pth")

fig=plt.figure()
plt.plot(accuracy_g)
plt.ylabel('accuracy')
plt.xlabel('epochs')
plt.savefig('accuracy.png')