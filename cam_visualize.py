import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
from tqdm import tqdm
import shutil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-v", action='store_true', help='whether it is to generate for validation set')
args = parser.parse_args()

for_validation = args.v
if for_validation:
    img_path = 'Dataset/2.validation/img'
    mask_path = 'Dataset/2.validation/mask'
    cam_path = './validation_out_cam'
    heatmap_path = "./validation_heatmap"
    assert os.path.exists(cam_path), "The cam for validation has not been generated!"
else:
    img_path = 'Dataset/3.testing/img'
    mask_path = 'Dataset/3.testing/mask'
    cam_path = './test_out_cam'
    heatmap_path = "./test_heatmap"
    assert os.path.exists(cam_path), "The cam for testing has not been generated!"

image_names = os.listdir(img_path)
npy_names = os.listdir(cam_path)

if not os.path.exists(heatmap_path):
    os.mkdir(heatmap_path)
else:
    shutil.rmtree(heatmap_path)
    os.mkdir(heatmap_path)

for i in tqdm(range(len(image_names))):
    cam = np.load(os.path.join(cam_path, npy_names[i]), allow_pickle=True)
    im = cv2.imread(os.path.join(img_path, image_names[i]))
    mask = cv2.imread(os.path.join(mask_path, image_names[i]))

    heatmap = (cam[0] * 255).astype(np.uint8)
    heatmap = np.expand_dims(heatmap,axis=2)
    heatmap_img = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    tumor = cv2.addWeighted(heatmap_img, 0.7, im, 0.3, 0)

    heatmap = (cam[1] * 255).astype(np.uint8)
    heatmap = np.expand_dims(heatmap,axis=2)
    heatmap_img = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    stroma = cv2.addWeighted(heatmap_img, 0.7, im, 0.3, 0)

    heatmap = (cam[2] * 255).astype(np.uint8)
    heatmap = np.expand_dims(heatmap,axis=2)
    heatmap_img = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    normal = cv2.addWeighted(heatmap_img, 0.7, im, 0.3, 0)

    plt.figure(i)
    plt.subplot(2,2,1)
    plt.imshow(cv2.cvtColor(tumor, cv2.COLOR_BGR2RGB))
    plt.title('tumor')
    plt.subplot(2,2,2)
    plt.imshow(cv2.cvtColor(stroma, cv2.COLOR_BGR2RGB))
    plt.title('stroma')
    plt.subplot(2,2,3)
    plt.imshow(cv2.cvtColor(normal, cv2.COLOR_BGR2RGB))
    plt.title('normal')
    plt.subplot(2,2,4)
    plt.imshow(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))
    plt.title('original image')
    plt.savefig(f'{heatmap_path}/{i:02d}.png')
    plt.close()