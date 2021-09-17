import numpy as np
import cv2
import os

img_path = 'Dataset/2.validation/img'
image_names = os.listdir(img_path)
cam_path = 'out_cam'
npy_names = os.listdir(cam_path)

for i in range(len(image_names)):
    cam = np.load(os.path.join(cam_path, npy_names[i]), allow_pickle=True)
    im = cv2.imread(os.path.join(img_path, image_names[i]))

    heatmap = (cam[0] * 255).astype(np.uint8)
    heatmap = np.expand_dims(heatmap,axis=2)
    heatmap_img = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    fin = cv2.addWeighted(heatmap_img, 0.7, im, 0.3, 0)
    # cv2.imshow('result.png',fin)
    cv2.imwrite('result.png',fin)
    # cv2.waitKey(0)
    break