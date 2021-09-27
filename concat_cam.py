import numpy as np
import os
from matplotlib import pyplot as plt
from tqdm import tqdm
import png
from PIL import Image


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


model_names = ['secondphase_9632_ep10', 'secondphase_12856_ep10', 'secondphase_16456_last']
out_path = "train_pseudomask"
if not os.path.exists(out_path):
    os.mkdir(out_path)
train_path = "Dataset/1.training"
for file in tqdm(os.listdir(train_path)):
    fileindex = file[:-4]
    cam_total = None
    for model_name in model_names:
        cam_path = f'ensemble_candidates/{model_name}_cam_nonorm'
        cam_score = np.load(os.path.join(cam_path, f'{fileindex}.npy'), allow_pickle=True).astype(np.float32)
        if cam_total is None:
            cam_total = cam_score
        else:
            assert cam_total.size == cam_score.size
            cam_total += cam_score

    result_label = np.argmax(cam_total, axis=0).astype(np.uint8)
    np.save(f'{out_path}/{fileindex}.npy', result_label)

# img_path = 'Dataset/2.validation/img'
# gt_path = 'Dataset/2.validation/mask'
# mask_path = 'Dataset/2.validation/background-mask'

# # sample 8 images for each model to compare
# visualize_pick = [0, 7, 8, 9, 31, 34, 35, 39]

# for i in tqdm(visualize_pick):
#     cam_total = None
#     for model_name in model_names:
#         cam_path = f'out_cam/{model_name}_cam'
#         cam_score = np.load(os.path.join(cam_path, f'{i:02d}.npy'), allow_pickle=True).astype(np.float32)
#         if cam_total is None:
#             cam_total = cam_score
#         else:
#             assert cam_total.size == cam_score.size
#             cam_total += cam_score

#     mask = np.asarray(Image.open(mask_path + f'/{i:02d}.png')) 
#     groundtruth = np.asarray(Image.open(gt_path + f'/{i:02d}.png'))
#     cam = np.argmax(cam_total, axis=0).astype(np.uint8)
#     score = get_mIOU(mask, groundtruth, cam)
#     palette = [(0, 64, 128), (64, 128, 0), (243, 152, 0), (255, 255, 255)]
#     with open(f'{i:02d}.png', 'wb') as f:
#         w = png.Writer(cam.shape[1], cam.shape[0], palette=palette, bitdepth=8)
#         w.write(f, cam)

#     cam[mask == 1] = 3
#     with open(f'{i:02d}_1.png', 'wb') as f:
#         w = png.Writer(cam.shape[1], cam.shape[0],palette=palette, bitdepth=8)
#         w.write(f, cam)

#     plt.figure(i)
#     im = plt.imread(f'{i:02d}.png')
#     im_mask = plt.imread(f'{i:02d}_1.png')
#     gt = plt.imread(gt_path + f'/{i:02d}.png')
#     origin = plt.imread(img_path + f'/{i:02d}.png')

#     plt.figure(i, figsize=(40, 40))
#     plt.subplot(2, 2, 1)
#     plt.imshow(im)
#     plt.title(f'cam, mIOU = {score:.2f}')
#     plt.subplot(2, 2, 2)
#     plt.imshow(gt)
#     plt.title('groundtruth')
#     plt.subplot(2, 2, 3)
#     plt.imshow(origin)
#     plt.title('origin image')
#     plt.subplot(2, 2, 4)
#     plt.imshow(im_mask)
#     plt.title('cam with background mask')

#     if not os.path.exists('ensemble_heatmap'):
#         os.mkdir('ensemble_heatmap')
#     plt.savefig(f'ensemble_heatmap/{i:02d}.png')
#     plt.close()
