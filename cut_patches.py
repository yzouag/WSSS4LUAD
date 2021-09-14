import os
import numpy as np
from PIL import Image, ImagePalette
from patchify import patchify
import argparse
from tqdm import tqdm

def cropImage(imfile, c, count, threshold = 0.5):
    full_path = './Dataset/1.training/' + imfile
    im = Image.open(full_path)
    im_arr = np.asarray(im)
    patches = patchify(im_arr, (56, 56, 3), step = 28)
    # print(patches.shape)
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            if checkProportion(patches[i, j, 0, : , :, :], threshold):
                result = Image.fromarray(np.uint8(patches[i, j, 0, : , :, :]))
                result.save("./train_single_patches/image" + str(count) + "_" + str(i) + str(j) + '_' + str(c) + '.png')

def checkProportion(im_arr, threshold = 0.5):
    # assert len(im_arr.shape) == 3, "The imput image must have 3D shape!"
    # assert im_arr.shape[2] == 3, "Our input image has to be RGB type!"

    white = 0 
    other = 0  
    for i in range(im_arr.shape[0]):
        for j in range(im_arr.shape[1]):
            if np.sum(im_arr[i][j]) > 600:
                white += 1
            else:
                other += 1

    ratio = white / (other + white)
    if ratio < threshold:
        return True

    return False

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "-threshold", type=float, default=0.5, required=False, help="The threshold to use to eliminate images with white proportions")
    args = parser.parse_args()
    threshold = args.t
    # print(threshold)

    if not os.path.exists("./train_single_patches"):
        os.mkdir("./train_single_patches")

    single_dict = {"[1, 0, 0]": 0, "[0, 1, 0]": 1, "[0, 0, 1]": 2}
    count = 0
    for file in tqdm(os.listdir('./Dataset/1.training')):
        category = file.split('.')[0][-9:]
        if category in single_dict:
            count += 1
            c = single_dict[category]
            cropImage(file, c, count, threshold)

