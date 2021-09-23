import shutil
from PIL import Image
import argparse
from tqdm import tqdm
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
import dataset
import network
import torch
from math import inf
import os
import json
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

parser = argparse.ArgumentParser()
parser.add_argument("-v", action='store_true',
                    help='whether it is to generate for validation set')
parser.add_argument("-side", default=96, type=int, required=False)
parser.add_argument("-stride", default=32, type=int, required=False)
parser.add_argument("-m", default="model_last", type=str,
                    required=True, help="model name")
args = parser.parse_args()

for_validation = args.v
side_length = args.side
stride = args.stride
model_name = args.m

if for_validation:
    out_cam = "./validation_out_cam"
    dataset_path = "./Dataset/2.validation/img"
else:
    out_cam = "./train_out_cam"
    dataset_path = "./Dataset/1.training"

if not os.path.exists(out_cam):
    os.mkdir(out_cam)
else:
    shutil.rmtree(out_cam)
    os.mkdir(out_cam)

net = network.ResNetCAM()
path = "./modelstates/" + model_name + ".pth"
pretrained = torch.load(path)['model']
pretrained = {k[7:]: v for k, v in pretrained.items()}
pretrained['fc1.weight'] = pretrained['fc1.weight'].unsqueeze(-1).unsqueeze(-1).to(torch.float64)
pretrained['fc2.weight'] = pretrained['fc2.weight'].unsqueeze(-1).unsqueeze(-1).to(torch.float64)

net.load_state_dict(pretrained)
print(f'Model loaded from {path} Successfully')
net.cuda()
net.eval()

onlineDataset = dataset.OnlineDataset(dataset_path, transform=transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
    patch_size=side_length,
    stride=stride)

print("Dataset", len(onlineDataset))
onlineDataloader = DataLoader(onlineDataset, batch_size=1, drop_last=False)

with open('result.json') as f:
    big_labels = json.load(f)

for im_path, im_list, position_list in tqdm(onlineDataloader):
    print(im_path[0])
    break
    orig_img = np.asarray(Image.open(im_path[0]))
    def tocamlist(im):
        im = im.cuda()
        cam_scores = net(im)
        # expected shape is batch_size * channel * h * w
        cam_scores = F.interpolate(cam_scores, (side_length, side_length),
                                   mode='bilinear', align_corners=False)[0].detach().cpu().numpy()
        return cam_scores

    # numofimgs, channels, length, lengths
    cam_list = list(map(tocamlist, im_list))

    # merge crops
    sum_cam = np.zeros((3, orig_img.shape[0], orig_img.shape[1]))
    sum_counter = np.zeros_like(sum_cam)
    for i in range(len(cam_list)):
        y, x = position_list[i][0], position_list[i][1]
        crop = cam_list[i]
        sum_cam[:, y:y+side_length, x:x+side_length] += crop
        sum_counter[:, y:y+side_length, x:x+side_length] += 1
    sum_counter[sum_counter < 1] = 1

    sum_cam = sum_cam / sum_counter
    # cam_max = np.max(sum_cam, (1,2), keepdims=True)
    # cam_min = np.min(sum_cam, (1,2), keepdims=True)
    # sum_cam[sum_cam < cam_min+1e-5] = 0
    # norm_cam = (sum_cam-cam_min) / (cam_max - cam_min + 1e-5)
    big_label = big_labels[im_path[0][]]
    for i in range(3):
        if big_label[i] == 0:
            sum_cam[i, :, :] = -inf
    result_label = sum_cam.argmax(axis=0)

    if out_cam is not None:
        if not os.path.exists(out_cam):
            os.makedir(out_cam)
        np.save(os.path.join(out_cam, im_path[0].split('/')[-1].split('.')[0] + '.npy'), result_label)