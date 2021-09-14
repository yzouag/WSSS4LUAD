import torch
import model
import dataset
import os
from torch.utils.data import DataLoader
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from dataset import RandomCrop, ToTensor, CenterCrop
from tqdm import tqdm
import train_loss


batch_size = 1
net = model.VNet().cuda()

m = 'model.pth'
Dir = "/home/yyubm/SwinTR"

pretrained_dict = torch.load("./" + m)['model']

net.load_state_dict(pretrained_dict)
net.cuda()
net.eval()
test_file = np.load('test_file.npy')
Dataset = dataset.LungDataset(Dir, test_file, transform=transforms.Compose([
CenterCrop((4, 48, 96, 96)),
ToTensor()]))

print("Dataset", len(Dataset))
Dataloader = DataLoader(Dataset, batch_size=batch_size, num_workers=batch_size * 2)
count = 0
correct = 0
average_dice = []

for data in Dataloader:
    count += 1
    img = data['image'].float().cuda()
    label = data['label'].cuda()
    segmentation = data['segment'].cuda()
    segment_a = segmentation[0,0].cpu().numpy()
    segment_b = segmentation[0,1].cpu().numpy()
    # print("segment_a's shape is:", segment_a.shape)

    prediction, seg_a, seg_b = net(img)
    seg_a = torch.argmax(seg_a, axis=1)[0].cpu().numpy()
    # print("pridict shape is:", seg_a.shape)
    seg_b = torch.argmax(seg_b, axis=1)[0].cpu().numpy()
    dice_a, hd95_a = train_loss.eval_metric(seg_a, segment_a)
    dice_b, hd95_b = train_loss.eval_metric(seg_b, segment_b)
    average_dice.append(dice_a)
    average_dice.append(dice_b)
    print("dice scores:", dice_a, dice_b)
    print("prediction: ", prediction)
    level = int(prediction.detach().argmax(dim = 1)[0].item())
    print("level: ", level)
    print("label: ", label.item())
    if level == label:
        correct += 1

accuracy = correct / count
print("average_dice is:", sum(average_dice)/len(average_dice))
print('Testing performance is: accuracy : %f' % (accuracy))