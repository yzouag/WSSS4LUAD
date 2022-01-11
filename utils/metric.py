import os
from PIL import Image
import numpy as np
from multiprocessing import Array, Process
from utils.util import chunks
import random

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
            score += intersection / union
    num_cluster = len(np.unique(real))
    return score / num_cluster


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


def get_overall_valid_score(
    pred_image_path, groundtruth_path, num_workers=5, mask_path=None
):
    """
    get the scores with validation groundtruth, the background will be masked out
    and return the score for all photos

    Args:
        pred_image_path (str): the prediction require to test, npy format
        groundtruth_path (str): groundtruth images, png format
        num_workers (int): number of process in parallel, default is 5.
        mask_path (str): the white background, png format

    Returns:
        float: the mIOU score
    """
    image_names = map(lambda x: x.split('.')[0], os.listdir(pred_image_path))
    random.shuffle(image_names)
    image_list = chunks(image_names, num_workers)

    def f(intersection, union, image_list):
        gt_list = []
        pred_list = []

        for im_name in image_list:
            cam = np.load(os.path.join(pred_image_path, f"{im_name}.npy"), allow_pickle=True).astype(np.uint8).reshape(-1)
            groundtruth = np.asarray(Image.open(groundtruth_path + f"/{im_name}.png")).reshape(-1)
            
            if mask_path:
                mask = np.asarray(Image.open(mask_path + f"/{im_name}.png")).reshape(-1)
                cam = cam[mask == 0]
                groundtruth = groundtruth[mask == 0]
            
            gt_list.extend(groundtruth)
            pred_list.extend(cam)

        pred = np.array(pred_list)
        real = np.array(gt_list)
        for i in [0, 1, 2]:
            if i in pred:
                inter = sum(np.logical_and(pred == i, real == i))
                u = sum(np.logical_or(pred == i, real == i))
                intersection[i] += inter
                union[i] += u

    intersection = Array("d", [0, 0, 0])
    union = Array("d", [0, 0, 0])

    p_list = []
    for i in range(num_workers):
        p = Process(target=f, args=(intersection, union, image_list[i]))
        p.start()
        p_list.append(p)
    for p in p_list:
        p.join()

    eps = 1e-7
    class0 = intersection[0] / (union[0] + eps)
    class1 = intersection[1] / (union[1] + eps)
    class2 = intersection[2] / (union[2] + eps)
    return (class0 + class1 + class2) / 3
