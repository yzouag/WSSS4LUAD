import os
import numpy as np
from PIL import Image, ImagePalette

if not os.path.exists("./test_results"):
    os.mkdir("./test_results")

campath = "./test_out_cam"
for file in os.listdir("./Dataset/3.testing/background-mask"):
    fileindex = file.split('.')[0]
    print(file)
    mask = Image.open("./Dataset/3.testing/background-mask/"+ file)
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
    result.save("./test_results/" + file)