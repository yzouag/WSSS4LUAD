import os
import numpy as np
from PIL import Image
from patchify import patchify
import argparse
from multiprocessing import Pool
import shutil
from tqdm import tqdm
import torch
import network
from torchvision import transforms


def crop_train_image(file_info):
    imfile, count, threshold = file_info
    full_path = dataset_path + imfile
    im = Image.open(full_path)
    im_arr = np.asarray(im)
    patches = patchify(im_arr, (patch_shape, patch_shape, 3), step=stride)
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            sub_image = patches[i, j, 0, :, :, :]
            if is_valid_crop(sub_image, threshold):
                im_type = get_label_from_nn(sub_image)
                result = Image.fromarray(np.uint8(patches[i, j, 0, :, :, :]))
                result.save(cut_result_path + str(count) +
                            "_" + str(i) + str(j) + '_' + str(im_type) + '.png')


def get_label_from_nn(sub_image, f, upper=1, lower=-1):
    im_type = [0, 0, 0]
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    image = transform(sub_image)
    image = image.reshape((1, 3, 224, 224))
    with torch.no_grad():
        input = image.cuda()
        score = net(input)
        result = score.cpu().numpy().reshape(-1)
    f.write(str(result) + '\n')
    for i in range(3):
        if result[i] > upper:
            im_type[i] = 1
        elif result[i] < lower:
            im_type[i] = 0
        else:
            im_type[i] = -1
    return im_type


def crop_valid_image(origin_im, mask_im, count, threshold, cut_result_path):
    stack_image = np.concatenate((origin_im, mask_im.reshape(
        mask_im.shape[0], mask_im.shape[1], 1)), axis=2)
    patches = patchify(stack_image, (patch_shape, patch_shape, 4), step=stride)
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            sub_image = patches[i, j, 0, :, :, :3]
            if is_valid_crop(sub_image, groundtruth=True):
                label = patches[i, j, 0, :, :, 3]
                im_type = get_labels(label, threshold)
                result = Image.fromarray(np.uint8(sub_image))
                result.save(cut_result_path + '/image' + str(count) +
                            '_' + str(i) + str(j) + '_' + str(im_type) + '.png')


def is_valid_crop(im_arr, threshold=0.5, groundtruth=True):
    WHITE = 600
    if groundtruth:
        count = np.sum(im_arr == 3)
        if count/im_arr.size > threshold:
            return False
        else:
            return True
    else:
        white = np.sum(im_arr, axis=2) > WHITE
        if np.sum(white) / white.size > threshold:
            return False
        else:
            return True


def get_labels(label, threshold=0.3):
    pix_type, pix_count = np.unique(label, return_counts=True)
    im_type = [0, 0, 0, 0]
    for i in range(len(pix_type)):
        if pix_count[i] / label.size > threshold:
            im_type[pix_type[i]] = 1
    return im_type[:3]


def test_crop_accuracy(test_path):
    files = os.listdir(test_path)
    count = 0
    with open('haha.txt', 'a') as f:
        for image_name in files:
            full_path = os.path.join(test_path, image_name)
            label = image_name[-13:-4]
            groundtruth = [int(label[1]), int(label[4]), int(label[7])]
            image = np.asarray(Image.open(full_path))
            prediction = get_label_from_nn(image, f)
            for i in range(3):
                if groundtruth[i] == prediction[i]:
                    count += 1
    print(count/len(files)*3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "-threshold", type=float, default=0.5, required=False,
                        help="The threshold to use to eliminate images with white proportions")
    parser.add_argument("-shape", default=56, type=int)
    parser.add_argument("-stride", default=28, type=int)
    parser.add_argument("-d", "--dataset", default=1, type=int,
                        help="the crop dataset, 1.training, 2.validation, 3.testing", choices=[1, 2, 3])
    parser.add_argument("-test", action='store_true', help='take the test')
    args = parser.parse_args()

    model_path = 'modelstates/model_last.pth'
    model_param = torch.load(model_path)['model']
    net = network.ResNet()
    net.load_state_dict(model_param)
    print(f'Model loaded from {model_path}')
    net.cuda()
    net.eval()

    if args.test:
        valid = 'valid_single_patches'
        test_crop_accuracy(valid)
        exit()

    threshold = args.t
    patch_shape = args.shape
    stride = args.stride
    dataset = args.dataset

    if dataset == 1:
        dataset_path = 'Dataset/1.training'
        cut_result_path = "./train_single_patches"
    elif dataset == 2:
        valid_mask_path = 'Dataset/2.validation/mask'
        valid_origin_path = 'Dataset/2.validation/img'
        cut_result_path = "./valid_single_patches"
    else:
        dataset_path = 'Dataset/3.testing'
        cut_result_path = "./test_single_patches"

    if not os.path.exists(cut_result_path):
        os.mkdir(cut_result_path)
    else:
        shutil.rmtree(cut_result_path)
        os.mkdir(cut_result_path)

    if dataset == 1:
        p = Pool(processes=6)
        file_list = []
        count = 0
        for file in os.listdir(dataset):
            label = file.split('-')[-1][:-4]
            labels = [int(label[1]), int(label[4]), int(label[7])]
            file_list.append((file, count, threshold))
        tqdm.tqdm(p.imap(crop_train_image, file_list), total=len(file_list))

    if dataset == 2:
        image_names = os.listdir(valid_mask_path)
        count = 0
        for image in tqdm(image_names):
            count += 1
            origin_image_path = os.path.join(valid_origin_path, image)
            mask_image_path = os.path.join(valid_mask_path, image)
            origin_im = np.asarray(Image.open(origin_image_path))
            mask_im = np.asarray(Image.open(mask_image_path))
            crop_valid_image(origin_im, mask_im, count,
                             threshold, cut_result_path)
