from PIL import Image
from tqdm import tqdm
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
import dataset
import torch
from math import inf
import os
import shutil
import json

# IMPORTANT! Note we DO NOT use the norm in all cases.

def generate_cam(net, model_name, model_crop, batch_size, mode, resize):
    """
    generate the class activation map using the model pass into

    Args:
        net (torch.models): the classification model
        model_name (string): the name to store in ensemble_candidate
        model_crop (tuple): (side_length, stride)
        batch_size (int): batch to process the cam
        mode (string): three options, 'train', 'valid', 'test'
    """
    # out_path = 'train_pseudomask'
    
    # if not os.path.exists(out_path):
    #     os.mkdir(out_path)
    # else:
    #     shutil.rmtree(out_path)
    #     os.mkdir(out_path)
    net.cuda()
    net.eval()

    side_length, stride = model_crop

    if mode == 'train':
        dataset_path = 'Dataset/1.training'
    elif mode == 'valid':
        dataset_path = 'Dataset/2.validation/img'
    else:
        dataset_path = 'Dataset/3.testing/img'

    onlineDataset = dataset.OnlineDataset(dataset_path, 
        transform=transforms.Compose([
            transforms.Resize((resize,resize)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        patch_size=side_length,
        stride=stride
    )

    print("Dataset", len(onlineDataset))
    onlineDataloader = DataLoader(onlineDataset, batch_size=1, drop_last=False)

    with torch.no_grad():
        for im_path, im_list, position_list in tqdm(onlineDataloader):
            orig_img = np.asarray(Image.open(im_path[0]))
            interpolatex = side_length
            interpolatey = side_length
            if orig_img.shape[0] < side_length:
                interpolatex = orig_img.shape[0]
            if orig_img.shape[1] < side_length:
                interpolatey = orig_img.shape[1]

            im_list = torch.vstack(im_list)

            im_list = torch.split(im_list, batch_size)
            cam_list = []
            for ims in im_list:
                cam_scores = net(ims.cuda())
                cam_scores = F.interpolate(cam_scores, (interpolatex, interpolatey), mode='bilinear', align_corners=False).detach().cpu().numpy()
                cam_list.append(cam_scores)
            cam_list = np.concatenate(cam_list)

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

            if mode == 'train':
                big_label = np.array([int(im_path[0][-12]), int(im_path[0][-9]), int(im_path[0][-6])])
                for k in range(3):
                    if big_label[k] == 0:
                        norm_cam[k, :, :] = -inf

                result_label = np.argmax(norm_cam, axis=0).astype(np.uint8)

                if not os.path.exists('ensemble_candidates'):
                    os.mkdir('ensemble_candidates')

                if not os.path.exists(f'ensemble_candidates/{model_name}_cam'):
                    os.mkdir(f'ensemble_candidates/{model_name}_cam')
                
                resultpath = im_path[0].split('/')[-1].split('.')[0]
                np.save(f'ensemble_candidates/{model_name}_cam/{resultpath}.npy', norm_cam)
                # np.save(f'{out_path}/{resultpath}.npy', result_label)  ### why we need to save this??????

            if mode == 'valid':         
                with open('groundtruth.json') as f:
                    big_labels = json.load(f)
                big_label = big_labels[im_path[0][-6:]]
                
                for k in range(3):
                    if big_label[k] == 0:
                        norm_cam[k, :, :] = -inf
                result_label = norm_cam.argmax(axis=0)

                if not os.path.exists('valid_out_cam'):
                    os.mkdir('valid_out_cam')

                # if not os.path.exists(f'out_cam/{model_name}_cam_nonorm'):
                #     os.mkdir(f'out_cam/{model_name}_cam_nonorm')
                # np.save(f'out_cam/{model_name}_cam_nonorm/{im_path[0][-6:-4]}.npy', norm_cam)

                if not os.path.exists(f'valid_out_cam/{model_name}'):
                    os.mkdir(f'valid_out_cam/{model_name}')
                np.save(f'valid_out_cam/{model_name}/{im_path[0][-6:-4]}.npy', result_label)

            if mode == 'test':
                result_label = norm_cam.argmax(axis=0)

                folder = 'test_candidates'
                if not os.path.exists(folder):
                    os.mkdir(folder)

                if not os.path.exists(f'{folder}/{model_name}'):
                    os.mkdir(f'{folder}/{model_name}')
                np.save(f'{folder}/{model_name}/{im_path[0][-6:-4]}.npy', norm_cam)

