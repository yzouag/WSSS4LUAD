import numpy as np
import cv2

cam = np.load('cam_val/01.npy', allow_pickle=True).item()
im = cv2.imread('Dataset/2.validation/img/01.png')

heatmap = (cam[1] * 255).astype(np.uint8)
# heatmap = np.expand_dims(heatmap,axis=2)
heatmap_img = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

fin = cv2.addWeighted(heatmap_img, 0.7, im, 0.3, 0)
cv2.imshow('result.png',fin)
cv2.waitKey(0)