import os
import numpy as np
from PIL import Image
from patchify import patchify
import argparse
from tqdm import tqdm
from collections import Counter

def crop_image(origin_im, mask_im, count, threshold):
    stack_image = np.concatenate((origin_im, mask_im.reshape(mask_im.shape[0], mask_im.shape[1],1)),axis=2)
    patches = patchify(stack_image, (56,56,4), step=56)
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            sub_image = patches[i,j,0,:,:,:3]
            label = patches[i,j,0,:,:,3]
            im_type = checkProportion(label, threshold)
            if im_type == 3 or im_type == -1:
                continue
            result = Image.fromarray(np.uint8(sub_image))
            result.save("./valid_single_patches/image" + str(count) + "_" + str(i) + str(j) + '_' + str(im_type) + '.png')

def checkProportion(im_arr, threshold = 0.7):
    im_arr = list(im_arr.reshape(-1))
    count = Counter(im_arr)
    imtype, counts = count.most_common(1)[0]
    if counts / len(im_arr) > threshold:
        return imtype
    else:
        return -1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "-threshold", type=float, default=0.7, required=False, help="The threshold to use to eliminate images with white proportions")
    args = parser.parse_args()
    threshold = args.t

    if not os.path.exists("./valid_single_patches"):
        os.mkdir("./valid_single_patches")
    
    valid_mask_path = 'Dataset/2.validation/mask'
    valid_origin_path = 'Dataset/2.validation/img'
    image_names = os.listdir(valid_mask_path)
    
    count = 0
    for image in tqdm(image_names):
        count += 1
        origin_image_path = os.path.join(valid_origin_path, image)
        mask_image_path = os.path.join(valid_mask_path, image)
        origin_im = np.asarray(Image.open(origin_image_path))
        mask_im = np.asarray(Image.open(mask_image_path))
        crop_image(origin_im, mask_im, count, threshold)