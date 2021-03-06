import json
import os
import png
import numpy as np
from skimage.measure import label
from PIL import Image
import torch
from torchvision.transforms import transforms
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm
import network
from functools import partial


def get_neighbors(point, image, is_stroma):
    # if point at corner, rerturn false
    x, y = point
    if x == 0 or y == 0 or x == image.shape[0]-1 or y == image.shape[1]-1:
        return False
    # if one neighbor is blue, return False
    if is_stroma:
        if image[x-1, y] == 0 or image[x+1, y] == 0 or image[x, y-1] == 0 or image[x, y+1] == 0:
            return False
    # else, return true
    return True


def color_exclusion(index_and_image, save_dir):
    image_i, image = index_and_image

    palette = [(0, 64, 128), (64, 128, 0), (243, 152, 0), (255, 255, 255)]
    with open(f'{image_i:02d}.png', 'wb') as f:
        w = png.Writer(image.shape[1], image.shape[0], palette=palette, bitdepth=8)
        w.write(f, image)

    image_normal = np.zeros_like(image)
    image_normal[image == 2] = 1
    disconnect_image, disconnect_num = label(image_normal, connectivity=1, return_num=True)

    for i in range(1, disconnect_num+1):
        # rule 1: change small yellow blocks to its surrounding color
        if np.sum(disconnect_image == i) < 200:
            image[disconnect_image == i] = 1
            continue
        # rule 2: if one yellow area is circled by green, turn it to green
        x, y = np.where(disconnect_image == i)
        index = list(zip(x, y))
        only_green = True
        for point in index:
            if not get_neighbors(point, image, is_stroma=False):
                only_green = False
                break
        if only_green:
            image[disconnect_image == i] = 1

    image_stroma = np.zeros_like(image)
    image_stroma[image == 1] = 1
    disconnect_image, disconnect_num = label(image_stroma, connectivity=1, return_num=True)

    for i in range(1, disconnect_num+1):
        # rule 3: if one green area is circled by yellow, turn into yellow
        x, y = np.where(disconnect_image == i)
        index = list(zip(x, y))
        only_yellow = True
        for point in index:
            if not get_neighbors(point, image, is_stroma=True):
                only_yellow = False
                break
        if only_yellow:
            image[disconnect_image == i] = 2

    palette = [(0, 64, 128), (64, 128, 0), (243, 152, 0), (255, 255, 255)]
    with open(f'{save_dir}/{image_i:02d}_new.png', 'wb') as f:
        w = png.Writer(image.shape[1], image.shape[0], palette=palette, bitdepth=8)
        w.write(f, image)

def cut_patches(im, im_size, stride, transform):
    im_list = []
    h, w, _ = im.shape

    h_ = np.arange(0, h - im_size + 1, stride)
    if h % stride != 0:
        h_ = np.append(h_, h-im_size)
    w_ = np.arange(0, w - im_size + 1, stride)
    if w % stride != 0:
        w_ = np.append(w_, w - im_size)

    for i in h_:
        for j in w_:   	
            temp = Image.fromarray(np.uint8(im[i:i+im_size,j:j+im_size,:]))
            temp = transform(temp)
            im_list.append(temp)
    return torch.stack(im_list)


def predict_big_label(big_label_net, image_path, im_size, stride, threshold):
    result = {}
    big_label_net.cuda()
    big_label_net.eval()
    # load images and cut to small patches
    image_names = os.listdir(image_path)
    for image in tqdm(image_names):
        im = np.asarray(Image.open(os.path.join(image_path, image)))
        transform = transforms.Compose([
                    transforms.Resize((448, 448)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        im_list = cut_patches(im, im_size=im_size, stride=stride, transform=transform).cuda()
        
        label = [0, 0, 0]
        total_scores = []
        
        # pass each sub crops to the model, here combine 64 sub crops together as one crop to the network
        for ims in torch.split(im_list, 4):
            with torch.no_grad():
                ims.cuda()
                scores = torch.sigmoid(big_label_net(ims))
                total_scores.append(scores.cpu().numpy())
        scores = np.vstack(total_scores)

        # for each kind of cells
        for i in range(3):
            # if 1 of the crop is predicted as positive, the whole picture will be positive
            if sum(scores[: ,i] > threshold[i]) >= 1:
                label[i] = 1
        result[image] = label
    
    # store the result in prediction.json, the format is 
    with open('prediction.json','w') as f:
        json.dump(result, f)

def load_and_scale_cam(segment_cam_path, tumor_ratio=0.9, with_big_label=False):
    # load cam from the dir
    npy_list = os.listdir(segment_cam_path)
    cam_list = []
    for cam_npy in npy_list:
        cam = np.load(os.path.join(segment_cam_path, cam_npy))
        if with_big_label:
            with open('prediction.json') as json_file:
                prediction_labels = json.load(json_file)
            big_label = prediction_labels[f'{cam[0:2]}.png']
            cam = cam * big_label.reshape(3,1,1)
        # scale the score to prevent over emphasizing on tumor
        cam[0] = cam[0] * tumor_ratio
        # get the max score as its label
        cam = np.argmax(cam, axis=0).astype(np.uint8)
        cam_list.append((int(cam_npy[0:2]), cam))
    return cam_list


if __name__ == '__main__':
    threshold = [0.5, 0.5, 0.5]
    
    path = ''
    net = network.scalenet101_cam(structure_path='structures/scalenet101.json')
    pretrained = torch.load(path)['model']
    pretrained = {k[7:]: v for k, v in pretrained.items()}
    pretrained['fc1.weight'] = pretrained['fc1.weight'].unsqueeze(-1).unsqueeze(-1).to(torch.float64)
    pretrained['fc2.weight'] = pretrained['fc2.weight'].unsqueeze(-1).unsqueeze(-1).to(torch.float64)

    net.load_state_dict(pretrained)
    print(f'Model loaded from {path} Successfully')
    
    predict_big_label(net, 'Dataset/2.validation/img', 225, 80, threshold)
    segment_cam_path = 'WSSS_tumor/r2n_crop_cam_npy_160_320_val/'
    img_list = load_and_scale_cam(segment_cam_path, tumor_ratio=0.9, with_big_label=True)
    save_dir_name = 'r2n_val_new'
    process_map(partial(color_exclusion, save_dir=save_dir_name), img_list, max_workers=6)
