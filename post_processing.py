import png
import numpy as np
from skimage.measure import label
from PIL import Image
from tqdm.contrib.concurrent import process_map


def get_neighbors(point, image, is_stroma):
    # if point at corner, rerturn false
    x, y = point
    if x == 0 or y == 0 or x == image.shape[0]-1 or y == image.shape[1]-1:
        return False
    # if one neighbor is blue, return False
    if is_stroma:
        if image[x-1, y] == 0 or image[x+1, y] == 0 or image[x, y-1] == 0 or image[x, y+1] == 0:
            return False
    # else, return true
    return True


def post_process(image_i):
    image_path = f'r2n_test/{image_i:02d}.png'
    image = np.asarray(Image.open(image_path))

    palette = [(0, 64, 128), (64, 128, 0), (243, 152, 0), (255, 255, 255)]
    with open(f'{image_i:02d}.png', 'wb') as f:
        w = png.Writer(image.shape[1], image.shape[0],
                       palette=palette, bitdepth=8)
        w.write(f, image)

    image_normal = np.zeros_like(image)
    image_normal[image == 2] = 1
    disconnect_image, disconnect_num = label(
        image_normal, connectivity=1, return_num=True)

    for i in range(1, disconnect_num+1):
        # rule 1: change small yellow blocks to its surrounding color
        if np.sum(disconnect_image == i) < 200:
            image[disconnect_image == i] = 1
            continue
        # rule 2: if one yellow area is circled by green, turn it to green
        x, y = np.where(disconnect_image == i)
        index = list(zip(x, y))
        only_green = True
        for point in index:
            if not get_neighbors(point, image, is_stroma=False):
                only_green = False
                break
        if only_green:
            image[disconnect_image == i] = 1

    image_stroma = np.zeros_like(image)
    image_stroma[image == 1] = 1
    disconnect_image, disconnect_num = label(
        image_stroma, connectivity=1, return_num=True)

    for i in range(1, disconnect_num+1):
        # rule 3: if one green area is circled by yellow, turn into yellow
        x, y = np.where(disconnect_image == i)
        index = list(zip(x, y))
        only_yellow = True
        for point in index:
            if not get_neighbors(point, image, is_stroma=True):
                only_yellow = False
                break
        if only_yellow:
            image[disconnect_image == i] = 2

    palette = [(0, 64, 128), (64, 128, 0), (243, 152, 0), (255, 255, 255)]
    with open(f'r2n_test_new/{image_i:02d}_new.png', 'wb') as f:
        w = png.Writer(image.shape[1], image.shape[0],
                       palette=palette, bitdepth=8)
        w.write(f, image)


if __name__ == '__main__':
    process_map(post_process, range(80), max_workers=6)
