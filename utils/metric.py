import os
from PIL import Image
import numpy as np
from tqdm import trange, tqdm

def calculate_IOU(pred, real):
    """
    this test is on a single image, and the number of clusters are the number in groundtruth
    thus, if prediction has three classes but gt has 2, mean will only divide by 2

    Returns:
        float: mIOU score
    """
    score = 0
    # num_cluster = 0
    for i in [0, 1, 2]:
        if i in pred:
            # num_cluster += 1
            intersection = sum(np.logical_and(pred == i, real == i))
            union = sum(np.logical_or(pred == i, real == i))
            score += intersection/union
    num_cluster = len(np.unique(real))
    return score/num_cluster


def get_mIOU(mask, groundtruth, prediction):
    """
    in this mIOU calculation, the mask will be excluded
    """
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

def get_overall_valid_score(pred_image_path):
    """
    get the scores with validation groundtruth, the background will be masked out
    and return the score for all photos

    Args:
        pred_image_path (str): the prediction require to test, npy format
        groundtruth_path (str): groundtruth images, png format
        mask_path (str): the white background, png format

    Returns:
        float: the mIOU score
    """
    groundtruth_path = 'Dataset/2.validation/mask'
    mask_path = 'Dataset/2.validation/background-mask'
    iou_score = 0

    gt_list = []
    pred_list = []

    for i in range(40):
        mask = np.asarray(Image.open(mask_path + f'/{i:02d}.png')).reshape(-1)
        cam = np.load(os.path.join(pred_image_path, f'{i:02d}.npy'), allow_pickle=True).astype(np.uint8).reshape(-1)
        groundtruth = np.asarray(Image.open(groundtruth_path + f'/{i:02d}.png')).reshape(-1)
        pred = cam[mask==0]
        gt = groundtruth[mask==0]
        gt_list.extend(gt)
        pred_list.extend(pred)
    pred = np.array(pred_list)
    real = np.array(gt_list)
    for i in tqdm([0, 1, 2]):
        if i in pred:
            intersection = sum(np.logical_and(pred == i, real == i))
            union = sum(np.logical_or(pred == i, real == i))
            iou_score += intersection/union
    return iou_score/3
    
