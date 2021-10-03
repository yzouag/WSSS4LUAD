import json
from PIL import Image
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
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
# IMPORTANT! Note we DO NOT use the norm in all cases.
dataset_path = "./Dataset/3.testing/img"
model_name = ['secondphase_scalenet101_224_last'] # 'secondphase_scalenet101_224_last'
model_crop = [(96, 32)]

for i in range(len(model_name)):

    net = network.scalenet101_cam(structure_path='structures/scalenet101.json')
    path = "./modelstates/" + model_name[i] + ".pth"
    pretrained = torch.load(path)['model']
    pretrained = {k[7:]: v for k, v in pretrained.items()}
    pretrained['fc1.weight'] = pretrained['fc1.weight'].unsqueeze(-1).unsqueeze(-1).to(torch.float64)
    pretrained['fc2.weight'] = pretrained['fc2.weight'].unsqueeze(-1).unsqueeze(-1).to(torch.float64)

    net.load_state_dict(pretrained)
    print(f'Model loaded from {path} Successfully')
    net.cuda()
    net.eval()

    side_length = model_crop[i][0]
    stride = model_crop[i][1]
    onlineDataset = dataset.OnlineDataset(dataset_path, transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
        patch_size=side_length,
        stride=stride)

    onlineDataloader = DataLoader(onlineDataset, batch_size=1, drop_last=False)

    for im_path, im_list, position_list in tqdm(onlineDataloader):

        orig_img = np.asarray(Image.open(im_path[0]))

        im_list = torch.vstack(im_list)
        batch_size = 24
        im_list = torch.split(im_list, batch_size)
        cam_list = []
        for ims in im_list:
            cam_scores = net(ims.cuda())
            cam_scores = F.interpolate(cam_scores, (side_length, side_length), mode='bilinear', align_corners=False).detach().cpu().numpy()
            cam_list.append(cam_scores)
        cam_list = np.concatenate(cam_list)

        sum_cam = np.zeros((3, orig_img.shape[0], orig_img.shape[1]))
        sum_counter = np.zeros_like(sum_cam)
        for k in range(len(cam_list)):
            y, x = position_list[k][0], position_list[k][1]
            crop = cam_list[k]
            sum_cam[:, y:y+side_length, x:x+side_length] += crop
            sum_counter[:, y:y+side_length, x:x+side_length] += 1
        sum_counter[sum_counter < 1] = 1

        norm_cam = sum_cam / sum_counter

        result_label = norm_cam.argmax(axis=0)

        folder = 'test_candidates'
        if not os.path.exists(folder):
            os.mkdir(folder)

        if not os.path.exists(f'{folder}/{model_name[i]}'):
            os.mkdir(f'{folder}/{model_name[i]}')
        np.save(f'{folder}/{model_name[i]}/{im_path[0][-6:-4]}.npy', norm_cam)

        # if not os.path.exists(f'out_cam/{model_name[i]}'):
            # os.mkdir(f'out_cam/{model_name[i]}')
        # np.save(f'out_cam/{model_name[i]}/{im_path[0][-6:-4]}.npy', result_label)
