import os
os.environ['CUDA_VISIBLE_DEVICES']='3'
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
from math import inf

dataset_path = "./Dataset/1.training"

modellist = ['secondphase_scalenet152_last']
model_crop = [(96, 32)]
out_path = "train_pseudomask"
if not os.path.exists(out_path):
    os.mkdir(out_path)
else:
    shutil.rmtree(out_path)
    os.mkdir(out_path)

for i in range(len(modellist)):
    model_name = modellist[i]
    # net = network.ResNetCAM()
    net = network.scalenet152_cam(structure_path='structures/scalenet152.json')
    path = "./modelstates/" + model_name + ".pth"
    pretrained = torch.load(path)['model']
    pretrained = {k[7:] : v for k, v in pretrained.items()}
    pretrained['fc1.weight'] = pretrained['fc1.weight'].unsqueeze(-1).unsqueeze(-1).to(torch.float64)
    pretrained['fc2.weight'] = pretrained['fc2.weight'].unsqueeze(-1).unsqueeze(-1).to(torch.float64)

    net.load_state_dict(pretrained)
    print(f'Model loaded from {path} Successfully')
    net.cuda()
    net.eval()
    side_length = model_crop[i][0]
    stride = model_crop[i][1]
    onlineDataset = dataset.OnlineDataset(dataset_path, transform=transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
        patch_size=side_length,
        stride=stride
        )

    print("Dataset", len(onlineDataset))
    onlineDataloader = DataLoader(onlineDataset, batch_size=1, drop_last=False)

    for im_path, im_list, position_list in tqdm(onlineDataloader):

        orig_img = np.asarray(Image.open(im_path[0]))
        interpolatex = side_length
        interpolatey = side_length
        if orig_img.shape[0] < side_length:
            interpolatex = orig_img.shape[0]
        if orig_img.shape[1] < side_length:
            interpolatey = orig_img.shape[1]

        im_list = torch.vstack(im_list)
        batch_size = 16
        im_list = torch.split(im_list, batch_size)
        cam_list = []
        for ims in im_list:
            cam_scores = net(ims.cuda())
            # print(cam_scores.shape)
            cam_scores = F.interpolate(cam_scores, (interpolatex, interpolatey), mode='bilinear', align_corners=False).detach().cpu().numpy()
            # print(cam_scores.shape)
            cam_list.append(cam_scores)
        cam_list = np.concatenate(cam_list)
        # cam_list = np.stack(cam_list)
        # print("output:", cam_list.shape)
        # def tocamlist(im):
        #     im = im.cuda()
        #     cam_scores = net(im)
        #     cam_scores = F.interpolate(cam_scores, (interpolatex, interpolatey), mode='bilinear', align_corners=False)[0].detach().cpu().numpy()
        #     return cam_scores

        # cam_list = list(map(tocamlist, im_list))

        sum_cam = np.zeros((3, orig_img.shape[0], orig_img.shape[1]))
        sum_counter = np.zeros_like(sum_cam)
        for k in range(cam_list.shape[0]):
            y, x = position_list[k][0], position_list[k][1]
            crop = cam_list[k]
            sum_cam[:, y:y+side_length, x:x+side_length] += crop
            sum_counter[:, y:y+side_length, x:x+side_length] += 1
        sum_counter[sum_counter < 1] = 1

        norm_cam = sum_cam / sum_counter
        # cam_max = np.max(sum_cam, (1, 2), keepdims=True)
        # cam_min = np.min(sum_cam, (1, 2), keepdims=True)
        # sum_cam[sum_cam < cam_min+1e-5] = 0
        # norm_cam = (sum_cam-cam_min) / (cam_max - cam_min + 1e-5)

        big_label = np.array([int(im_path[0][-12]), int(im_path[0][-9]), int(im_path[0][-6])])
        for k in range(3):
            if big_label[k] == 0:
                norm_cam[k, :, :] = -inf

        result_label = np.argmax(norm_cam, axis=0).astype(np.uint8)

        if not os.path.exists('ensemble_candidates'):
            os.mkdir('ensemble_candidates')

        if not os.path.exists(f'ensemble_candidates/{model_name}_cam_nonorm'):
            os.mkdir(f'ensemble_candidates/{model_name}_cam_nonorm')
        resultpath = im_path[0].split('/')[-1].split('.')[0]
        np.save(f'ensemble_candidates/{model_name}_cam_nonorm/{resultpath}.npy', norm_cam)
        np.save(f'{out_path}/{resultpath}.npy', result_label)