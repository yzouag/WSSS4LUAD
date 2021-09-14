import torch
import network
import dataset
import os
from torch.utils.data import DataLoader
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm
import torchvision.models as models
# import train_loss


batch_size = 50
net = network.ResNet().cuda()

resnet101 = models.resnet101(pretrained=True) 
pretrained_dict =resnet101.state_dict()
model_dict = net.state_dict()
print("Pretrained:", len(pretrained_dict.keys()))
print("Model:", len(model_dict.keys()))
not_exist = [k for k in pretrained_dict.keys() if not k in model_dict.keys()]
print("not exists:", not_exist)
pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k in model_dict} 
print(len(pretrained_dict))
# model_dict.update(pretrained_dict) 
# net.load_state_dict(model_dict)

# net.load_state_dict(pretrained_dict)
# net.cuda()
# net.eval()

# Dataset = dataset.SingleLabelDataset("train_single_patches/", transform=transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomVerticalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))

# print("Dataset", len(Dataset))
# Dataloader = DataLoader(Dataset, batch_size=batch_size, num_workers=batch_size * 2)
# count = 0
# correct = 0
# average_dice = []

# for data in Dataloader:
#     count += 1
#     img = data['image'].float().cuda()
#     label = data['label'].cuda()
#     segmentation = data['segment'].cuda()
#     segment_a = segmentation[0,0].cpu().numpy()
#     segment_b = segmentation[0,1].cpu().numpy()
#     # print("segment_a's shape is:", segment_a.shape)

#     prediction, seg_a, seg_b = net(img)
#     seg_a = torch.argmax(seg_a, axis=1)[0].cpu().numpy()
#     # print("pridict shape is:", seg_a.shape)
#     seg_b = torch.argmax(seg_b, axis=1)[0].cpu().numpy()
#     dice_a, hd95_a = train_loss.eval_metric(seg_a, segment_a)
#     dice_b, hd95_b = train_loss.eval_metric(seg_b, segment_b)
#     average_dice.append(dice_a)
#     average_dice.append(dice_b)
#     print("dice scores:", dice_a, dice_b)
#     print("prediction: ", prediction)
#     level = int(prediction.detach().argmax(dim = 1)[0].item())
#     print("level: ", level)
#     print("label: ", label.item())
#     if level == label:
#         correct += 1

# accuracy = correct / count
# print("average_dice is:", sum(average_dice)/len(average_dice))
# print('Testing performance is: accuracy : %f' % (accuracy))