from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth, spectral_clustering
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from matplotlib import pyplot as plt
import os
import cv2
from itertools import permutations

def get_mask(mask_image):
    background_im = np.reshape(mask_image, (-1,3))
    mask = np.logical_not(np.sum(background_im, axis=1) > 0)
    return mask

def get_groundtruth(groundtruth_image):
    groundtruth_im = cv2.cvtColor(groundtruth_image,cv2.COLOR_BGR2RGB).reshape(-1,3)
    
    length,_ = groundtruth_im.shape
    groundtruth = np.empty(length,dtype=np.uint8)

    for i in range(length): # for each pixel, get its label
        if (groundtruth_im[i] == color[TUMOR]).all():
            groundtruth[i] = TUMOR
        elif (groundtruth_im[i] == color[STROMA]).all():
            groundtruth[i] = STROMA
        elif (groundtruth_im[i] == color[NORMAL]).all():
            groundtruth[i] = NORMAL
        else:
            groundtruth[i] = BACKGROUND
    return groundtruth

def get_window_smooth(labeled_image, window_size=3):
    """
    use a sliding window to get majority of pixel labels in the window
    white(background) pixels remain white, the majority only consider non-white
    labels.

    Args:
        labeled_image (np.array): the image with labels [0,1,2,3]
        window_size (int): the size of the sliding window. Default to 3.
    
    Return:
        [np.array]: the image after smoothing
    """
    prediction = np.copy(labeled_image)
    for x in range(window_size, labeled_image.shape[0]-window_size):
        for y in range(window_size, labeled_image.shape[1]-window_size):
            # if background, remain background
            if labeled_image[x, y] == BACKGROUND:
                continue
            # if not background, take the majority of non backgound
            color, freq = np.unique(labeled_image[x-window_size:x+window_size+1,
                                        y-window_size:y+window_size+1].reshape(-1), return_counts=True)
            a = list(zip(freq, color))
            a.sort(reverse=True, key=lambda x: x[0])
            if a[0][1] == BACKGROUND:
                prediction[x, y] = a[1][1]
            else:
                prediction[x, y] = a[0][1]
    return prediction

def get_prediction(original_image, clusters, method='kmeans', heatmap=None, smoothing=True):
    # parameters to set
    white = 2.7
    window_size = 3
    gaussian_k = (3,3)
    pow = 2
    threshold = 1e-2
    weight = 1

    h, w, _ = original_image.shape
    
    blur_im = np.reshape(cv2.GaussianBlur(original_image, gaussian_k, 0), (-1,3))
    scaler = MinMaxScaler()
    blur_im = scaler.fit_transform(blur_im)

    if heatmap is not None:
        heatmap = heatmap**pow
        heatmap[heatmap <= threshold] = 0
        heatmap = heatmap.reshape(-1,3)*weight
        blur_im = np.hstack([blur_im, heatmap])

    new_im = np.zeros(h*w)
    
    filter_list = []
    mark = 0
    correspond = {}

    for i in range(len(blur_im)):
        if sum(blur_im[i,0:3]) <= white:
            filter_list.append(blur_im[i])
            correspond[mark] = i
            mark += 1
        else:
            new_im[i] = BACKGROUND
    filter_list = np.stack(filter_list)

    # now the color is undetermined
    if method == 'kmeans':
        kmeans = KMeans(n_clusters=clusters, random_state=0).fit(filter_list)
        labels = kmeans.labels_
    elif method == 'meanshift':
        bandwidth = estimate_bandwidth(filter_list, quantile=0.1, n_samples=2000)
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        ms.fit(filter_list)
        labels = ms.labels_
    elif method == 'spectral':
        labels = spectral_clustering(filter_list, n_clusters=clusters, eigen_solver='arpack')

    for i in range(len(filter_list)):
        if labels[i] == 1:
            new_im[correspond[i]] = STROMA
        elif labels[i] == 0:
            new_im[correspond[i]] = TUMOR
        elif labels[i] == 2:
            new_im[correspond[i]] = NORMAL

    new_im = new_im.reshape((h, w))
    if smoothing:
        return get_window_smooth(new_im, window_size=window_size)
    else:
        return new_im

def calculate_IOU(pred, real):
    score = 0
    num_cluster = 0
    for i in [TUMOR, STROMA, NORMAL]:
        if i in pred:
            num_cluster += 1
            intersection = sum(np.logical_and(pred==i, real==i))
            union = sum(np.logical_or(pred==i, real==i))
            score += intersection/union
    return score/num_cluster

def get_mIOU(mask, groundtruth, prediction):
    prediction = np.reshape(prediction, (-1))
    length = len(prediction)

    after_mask_pred = []
    after_mask_true = []
    for i in range(length):
        if mask[i]:
            after_mask_true.append(groundtruth[i])
            after_mask_pred.append(prediction[i])

    after_mask_pred = np.array(after_mask_pred)
    after_mask_true = np.array(after_mask_true)

    pred = np.zeros(len(after_mask_pred))
    max_score = 0
    perms = list(permutations([0,1,2]))
    for perm in perms:
        for i in range(len(after_mask_pred)):
            for j in range(3):
                if after_mask_pred[i] == j:
                    pred[i] = perm[j]
            if after_mask_pred[i] == 3:
                pred[i] = 3
        score = calculate_IOU(pred, after_mask_true)
        if score > max_score:
            max_score = score
            match = perm
    return max_score, match

def set_color(prediction, match):
    h,w = prediction.shape
    prediction_image = np.zeros((h,w,3))

    for x in range(h):
        for y in range(w):
            if prediction[x, y] == BACKGROUND:
                prediction_image[x,y] = color[BACKGROUND]
            elif prediction[x, y] == NORMAL:
                prediction_image[x,y] = color[match[NORMAL]]
            elif prediction[x, y] == TUMOR:
                prediction_image[x,y] = color[match[TUMOR]]
            else:
                prediction_image[x,y] = color[match[STROMA]]
    return prediction_image

TUMOR = 0
STROMA = 1
NORMAL = 2
BACKGROUND = 3

color = {
    TUMOR: np.array([0, 64, 128],dtype=np.uint8),
    STROMA: np.array([64, 128, 0],dtype=np.uint8),
    NORMAL: np.array([243, 152, 0],dtype=np.uint8),
    BACKGROUND: np.array([255, 255, 255],dtype=np.uint8)
}


k = [2, 1, 1, 2, 2, 1, 2, 2, 2, 3, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2,
     2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 3, 3, 3, 2, 2, 2, 2]

if __name__ == "__main__":
    image_names = os.listdir('2.validation/img')
    method = 'kmeans'
    with_cam = '_cam'
    
    for i in range(30):
        im_path = os.path.join('2.validation/img', image_names[i])
        mask_path = os.path.join('2.validation/background-mask', image_names[i])
        groundtruth_path = os.path.join('2.validation/mask', image_names[i])
        heatmap_path = os.path.join('cam_val', image_names[i].replace('.png','.npy'))
        
        original_image = cv2.imread(im_path)
        mask_image = cv2.imread(mask_path)
        groundtruth_image = cv2.imread(groundtruth_path)
        heatmap = None
        if with_cam != '':
            heatmap = np.load(heatmap_path, allow_pickle=True)
            heatmap = np.stack(list(heatmap.tolist().values()), axis=2) # stack the heatmap information to a three channel image
        
        mask = get_mask(mask_image)
        groundtruth = get_groundtruth(groundtruth_image)
        prediction = get_prediction(original_image, clusters=k[i], method=method, heatmap=heatmap, smoothing=True)
        
        mIOU, match = get_mIOU(mask, groundtruth, prediction)

        prediction_image = set_color(prediction, match).astype(np.int32)

        with open(f'res_{method}{with_cam}/result.log', 'a') as f:
            f.write(f'image{i}, mIOU={mIOU}\n')
        
        figure_size = 15
        plt.figure(figsize=(figure_size, figure_size))
        plt.subplot(1, 3, 1), plt.imshow(original_image)
        plt.title('Original Image')
        plt.subplot(1, 3, 2), plt.imshow(prediction_image)
        plt.title(f'Segmented Image, mIOU = {mIOU:.3f}')
        plt.subplot(1, 3, 3), plt.imshow(cv2.cvtColor(groundtruth_image, cv2.COLOR_BGR2RGB))
        plt.title('groundtruth image')
        plt.savefig(f'res_{method}{with_cam}/{i}.png')
        plt.close()
