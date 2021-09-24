import numpy as np
import os
from matplotlib import pyplot as plt
from tqdm import tqdm
import png
from PIL import Image
import shutil
import argparse


def calculate_IOU(pred, real):
    score = 0
    num_cluster = 0
    for i in [0, 1, 2]:
        if i in pred:
            num_cluster += 1
            intersection = sum(np.logical_and(pred == i, real == i))
            union = sum(np.logical_or(pred == i, real == i))
            score += intersection/union
    return score/num_cluster


def get_mIOU(mask, groundtruth, prediction):
    prediction = np.reshape(prediction, (-1))
    groundtruth = groundtruth.reshape(-1)
    mask = mask.reshape(-1)
    print(prediction.shape)
    print(groundtruth.shape)
    print(mask.shape)
    length = len(prediction)

    after_mask_pred = []
    after_mask_true = []
    for i in range(length):
        if mask[i] == 0:
            after_mask_true.append(groundtruth[i])
            after_mask_pred.append(prediction[i])

    after_mask_pred = np.array(after_mask_pred)
    after_mask_true = np.array(after_mask_true)
    score = calculate_IOU(after_mask_pred, after_mask_true)
    return score


parser = argparse.ArgumentParser()
parser.add_argument("-v", action='store_true',help='whether it is to generate for validation set')
args = parser.parse_args()

for_validation = args.v
if for_validation:
    img_path = 'Dataset/2.validation/img'
    gt_path = 'Dataset/2.validation/mask'
    cam_path = './validation_out_cam'
    heatmap_path = "./validation_heatmap"
    mask_path = 'Dataset/2.validation/background-mask'
    assert os.path.exists(cam_path), "The cam for validation has not been generated!"
else:
    img_path = 'Dataset/3.testing/img'
    gt_path = 'Dataset/3.testing/mask'
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


for i in tqdm(range(30)):
    mask = np.asarray(Image.open(mask_path+f'/{i:02d}.png'))
    cam = np.load(os.path.join(cam_path, npy_names[i]), allow_pickle=True).astype(np.uint8)
    palette = [(0, 64, 128), (64, 128, 0), (243, 152, 0), (255, 255, 255)]
    with open(f'{i:02d}.png', 'wb') as f:
        w = png.Writer(cam.shape[1], cam.shape[0], palette=palette, bitdepth=8)
        w.write(f, cam)
    groundtruth = np.asarray(Image.open(gt_path+f'/{i:02d}.png'))
    score = get_mIOU(mask, groundtruth, cam)

    cam[mask == 1] = 3
    with open(f'{i:02d}_1.png', 'wb') as f:
        w = png.Writer(cam.shape[1], cam.shape[0], palette=palette, bitdepth=8)
        w.write(f, cam)

    plt.figure(i)
    im = plt.imread(f'{i:02d}.png')
    im_mask = plt.imread(f'{i:02d}_1.png')
    gt = plt.imread(gt_path + f'/{i:02d}.png')
    origin = plt.imread(img_path + f'/{i:02d}.png')

    plt.figure(i, figsize=(40, 40))
    plt.subplot(2, 2, 1)
    plt.imshow(im)
    plt.title(f'cam, mIOU = {score:.2f}')
    plt.subplot(2, 2, 2)
    plt.imshow(gt)
    plt.title('groundtruth')
    plt.subplot(2, 2, 3)
    plt.imshow(origin)
    plt.title('origin image')
    plt.subplot(2, 2, 4)
    plt.imshow(im_mask)
    plt.title('cam with background mask')

    plt.savefig(f'heatmap/{i:02d}.png')
    plt.close()

# heatmap = ((cam/3)*255).astype(np.uint8)
# heatmap_img = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
# img = cv2.addWeighted(heatmap_img, 0.7, im, 0.3, 0)
# cv2.imshow('result',heatmap_img)
# cv2.waitKey(0)
# for i in tqdm(range(len(image_names))):
#     cam = np.load(os.path.join(cam_path, npy_names[i]), allow_pickle=True)
#     im = cv2.imread(os.path.join(img_path, image_names[i]))
#     mask = cv2.imread(os.path.join(mask_path, image_names[i]))

#     heatmap = (cam[0] * 255).astype(np.uint8)
#     heatmap = np.expand_dims(heatmap,axis=2)
#     heatmap_img = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
#     tumor = cv2.addWeighted(heatmap_img, 0.7, im, 0.3, 0)

#     heatmap = (cam[1] * 255).astype(np.uint8)
#     heatmap = np.expand_dims(heatmap,axis=2)
#     heatmap_img = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
#     stroma = cv2.addWeighted(heatmap_img, 0.7, im, 0.3, 0)

#     heatmap = (cam[2] * 255).astype(np.uint8)
#     heatmap = np.expand_dims(heatmap,axis=2)
#     heatmap_img = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
#     normal = cv2.addWeighted(heatmap_img, 0.7, im, 0.3, 0)

#     plt.figure(i)
#     plt.subplot(2,2,1)
#     plt.imshow(cv2.cvtColor(tumor, cv2.COLOR_BGR2RGB))
#     plt.title('tumor')
#     plt.subplot(2,2,2)
#     plt.imshow(cv2.cvtColor(stroma, cv2.COLOR_BGR2RGB))
#     plt.title('stroma')
#     plt.subplot(2,2,3)
#     plt.imshow(cv2.cvtColor(normal, cv2.COLOR_BGR2RGB))
#     plt.title('normal')
#     plt.subplot(2,2,4)
#     plt.imshow(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))
#     plt.title('original image')
#     plt.savefig(f'heatmap1/{i:02d}.png')
#     plt.show()
#     plt.close()
#     break