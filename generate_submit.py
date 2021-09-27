import os
import numpy as np
from PIL import Image
import argparse
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("-v", action='store_true', help='whether it is to generate for validation set')
args = parser.parse_args()

for_validation = args.v
if for_validation:
    out_path = "./validation_results"
    campath = "./out_cam/secondphase_scalenet101_last"
    background_path = "./Dataset/2.validation/background-mask/"
    assert os.path.exists(campath), "the cam for validation has not been generated!"
else:
    out_path = "./test_results"
    campath = "./test_out_cam"
    background_path = "./Dataset/3.testing/background-mask/"
    assert os.path.exists(campath), "the cam for test has not been generated!"

if not os.path.exists(out_path):
    os.mkdir(out_path)
else:
    shutil.rmtree(out_path)
    os.mkdir(out_path)

for file in os.listdir(background_path):
    fileindex = file.split('.')[0]
    print(file)
    mask = Image.open(background_path + file)
    mask_array = np.asarray(mask)
    i, j = mask_array.shape
    result_array = np.load(os.path.join(campath, fileindex+".npy"))
    # print(result_array[0][0] == [0, 0, 0])
    # if file == "00.png":
    #     print(result_array.shape)
    #     print(mask_array.shape)
    #     print(type(i))
    # print(mask_array)
    for a in range(i):
        for b in range(j):
            if mask_array[a][b] == 1:
                result_array[a][b] = 3


    result = Image.fromarray(np.uint8(result_array))
    result.save(os.path.join(out_path, file))