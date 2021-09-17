import os
# os.environ['CUDA_VISIBLE_DEVICES']='1, 2, 3'
import torch
import network
import dataset
from torch.utils.data import DataLoader
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import argparse
from PIL import Image
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("-v", action='store_true', help='whether it is to generate for validation set')
parser.add_argument("-side", default=56, type=int, required=False)
parser.add_argument("-stride", default=28, type=int, required=False)
parser.add_argument("-m", default="model_last", type=str, required=False, help="model name")
args = parser.parse_args()

for_validation = args.v
side_length = args.side
stride = args.stride
model_name = args.m
if for_validation:
    out_cam = "./validation_out_cam"
    dataset_path = "./Dataset/2.validation/img"
else:
    out_cam = "./test_out_cam"
    dataset_path = "./Dataset/3.testing/img"
if not os.path.exists(out_cam):
    os.mkdir(out_cam)
else:
    shutil.rmtree(out_cam)
    os.mkdir(out_cam)

net = network.ResNetCAM()
path = "./modelstates/" + model_name + ".pth"
pretrained = torch.load(path)['model']
# print(pretrained.keys())
pretrained = {k[7:] : v for k, v in pretrained.items()}
# pretrained_modify['fc1.weight'] = pretrained_modify['fc1.weight'].unsqueeze(-1).unsqueeze(-1)
# pretrained_modify['fc2.weight'] = pretrained_modify['fc2.weight'].unsqueeze(-1).unsqueeze(-1)
pretrained['fc1.weight'] = pretrained['fc1.weight'].unsqueeze(-1).unsqueeze(-1).to(torch.float64)
pretrained['fc2.weight'] = pretrained['fc2.weight'].unsqueeze(-1).unsqueeze(-1).to(torch.float64)
# print(pretrained['fc2.bias'].type())

net.load_state_dict(pretrained)
print(f'Model loaded from {path} Successfully')
# torch.save({"model": net.state_dict()}, "./model_last.pth")
net.cuda()
net.eval()


onlineDataset = dataset.OnlineDataset(dataset_path, transform=transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
    patch_size = side_length,
    stride = stride)

print("Dataset", len(onlineDataset))
onlineDataloader = DataLoader(onlineDataset, batch_size=1, drop_last=False)

for im_path, im_list, position_list in tqdm(onlineDataloader):
    orig_img = np.asarray(Image.open(im_path[0]))

    def tocamlist(im):
        
        im = im.cuda()
        cam_scores = net(im)
        # expected shape is batch_size * channel * h * w
        cam_scores = F.interpolate(cam_scores, (side_length, side_length), mode='bilinear', align_corners=False)[0].detach().cpu().numpy()
        return cam_scores

    # numofimgs, channels, length, lengths
    cam_list = list(map(tocamlist, im_list))
    # print(len(cam_list))

    # merge crops
    sum_cam = np.zeros((3, orig_img.shape[0], orig_img.shape[1]))
    sum_counter = np.zeros_like(sum_cam)
    for i in range(len(cam_list)):
        y, x = position_list[i][0], position_list[i][1]
        crop = cam_list[i]
        # print("i is :",i)
        # print(f"y: {y}, x: {x}")
        sum_cam[:, y:y+side_length, x:x+side_length] += crop
        sum_counter[:, y:y+side_length, x:x+side_length] += 1
    sum_counter[sum_counter < 1] = 1

    sum_cam = sum_cam / sum_counter
    # cam_max = np.max(sum_cam, (1,2), keepdims=True)
    # cam_min = np.min(sum_cam, (1,2), keepdims=True)
    # sum_cam[sum_cam < cam_min+1e-5] = 0
    # norm_cam = (sum_cam-cam_min) / (cam_max - cam_min + 1e-5)
    result_label = sum_cam.argmax(axis=0)

    if out_cam is not None:
        if not os.path.exists(out_cam):
            os.makedir(out_cam)
        np.save(os.path.join(out_cam, im_path[0].split('/')[-1].split('.')[0] + '.npy'), result_label)