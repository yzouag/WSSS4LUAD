import os
from PIL import Image
import numpy as np
from multiprocessing import Array, Process
from utils.util import chunks

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

def get_overall_valid_score(pred_image_path, num_workers=5):
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
    l = np.random.permutation(40)
    image_list = chunks(l, num_workers)

    def f(intersection, union, image_list):
        groundtruth_path = 'Dataset/2.validation/mask'
        mask_path = 'Dataset/2.validation/background-mask'
        gt_list = []
        pred_list = []
        
        for i in image_list:
            mask = np.asarray(Image.open(mask_path + f'/{i:02d}.png')).reshape(-1)
            cam = np.load(os.path.join(pred_image_path, f'{i:02d}.npy'), allow_pickle=True).astype(np.uint8).reshape(-1)
            groundtruth = np.asarray(Image.open(groundtruth_path + f'/{i:02d}.png')).reshape(-1)
            pred = cam[mask==0]
            gt = groundtruth[mask==0]
            gt_list.extend(gt)
            pred_list.extend(pred)
        
        pred = np.array(pred_list)
        real = np.array(gt_list)
        for i in [0, 1, 2]:
            if i in pred:
                inter = sum(np.logical_and(pred == i, real == i))
                u = sum(np.logical_or(pred == i, real == i))
                intersection[i] += inter
                union[i] += u

    intersection = Array('d', [0,0,0])
    union = Array('d', [0,0,0])

    p_list = []
    for i in range(num_workers):
        p = Process(target=f, args=(intersection, union, image_list[i]))
        p.start()
        p_list.append(p)
    for p in p_list:
        p.join()
    class0 = intersection[0]/(union[0]+0.000001)
    class1 = intersection[1]/(union[1]+0.000001)
    class2 = intersection[2]/(union[2]+0.000001)
    return (class0 + class1 + class2)/3

