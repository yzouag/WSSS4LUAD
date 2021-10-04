from multiprocessing.context import Process
import os
import numpy as np
from PIL import Image
from patchify import patchify
from multiprocessing import Manager
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import torch
import network
from torchvision import transforms
from math import ceil
import json


def crop_train_image(file_info):
    """This function crop the training images and save sub images in the `cut_result_path` folder."""
    imfile, count, threshold, labels, cut_result_path, patch_shape, stride = file_info
    im = Image.open(imfile)
    im_arr = np.asarray(im)
    if im_arr.shape[0] < patch_shape or im_arr.shape[1] < patch_shape:
        return
    patches = patchify(im_arr, (patch_shape, patch_shape, 3), step=stride)
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            sub_image = patches[i, j, 0, :, :, :]
            if is_valid_crop(sub_image, threshold, groundtruth=False):
                result = Image.fromarray(np.uint8(sub_image))
                result.save(os.path.join(cut_result_path, str(
                    count) + "_" + str(i) + str(j) + str(labels) + '.png'))


def crop_valid_image(origin_im, mask_im, index, threshold, white_threshold, cut_result_path):
    """This function crop the validation images and save the sub images in the `cut_result_path` folder."""
    stack_image = np.concatenate((origin_im, mask_im.reshape(
        mask_im.shape[0], mask_im.shape[1], 1)), axis=2)
    patches = patchify(stack_image, (patch_shape, patch_shape, 4), step=stride)
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            sub_image = patches[i, j, 0, :, :, :3]
            label = patches[i, j, 0, :, :, 3]
            if is_valid_crop(label, white_threshold, groundtruth=True):
                im_type = get_labels(label, threshold)
                result = Image.fromarray(np.uint8(sub_image))
                result.save(cut_result_path + '/image' + index +
                            '_' + str(i) + str(j) + '_' + str(im_type) + '.png')


def is_valid_crop(im_arr, threshold=0.9, groundtruth=True):
    """
    This function check whether the cropped sub images are valid for both the training and validation set.
    All pixels with sum value over 600 are considered as white and we remove all subcrops with white porpotion
    greater than the given threshold.
    """
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


def get_labels(label, threshold):
    """
    This function generates the one-hot label for the given 2D groundtruth by the given activation threshold.
    """
    pix_type, pix_count = np.unique(label, return_counts=True)
    im_type = [0, 0, 0, 0]
    for i in range(len(pix_type)):
        if pix_count[i] / label.size > threshold:
            im_type[pix_type[i]] = 1
    return im_type[:3]


def generate_image_label_score(test_path, save_name, num_workers=3, batch_size=64, is_new=True):
    """
    Mainly the multi-process managements, the key function is predict_image_score.
    """
    files = os.listdir(test_path)
    image_chunks = chunks(files, num_workers, -1)

    with Manager() as manager:
        L = manager.list()
        processes = []
        for i in range(num_workers):
            p = Process(target=predict_image_score, args=(
                L, image_chunks[i], test_path, batch_size, is_new))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        if not os.path.exists('image_label_score'):
            os.mkdir('image_label_score')
        np.save(f'image_label_score/{save_name}.npy', list(L))


def test_crop_accuracy(score_path, big_labels_path, min_amount):
    """
    Based on the image name and prediction scores, generate the best lower and upper confident threshold.
    """
    scores = np.load(score_path, allow_pickle=True)
    # big_labels = np.load(big_labels_path, allow_pickle=True)
    gt = []
    pred = []
    for i in range(len(scores)):
        pred.append(scores[i][1])
        label = scores[i][0][-13:-4]
        gt.append([int(label[1]), int(label[4]), int(label[7])])
    pred = np.stack(pred)
    gt = np.stack(gt)

    # get big image label, exclude impossible labels
    # for each label:
    #   get all images > upper, mark as 1, and their true label
    #   get all images < lower, mark as 0, and their true label
    #   if images predict 1 and correct / images predict 1 > 0.95, the threshold is confident enough
    #   same for lower bound
    # record lower and upper bound
    threshold = {}
    for i in tqdm(range(3)):
        lower = 0
        upper = 0
        # min_amount = 100
        best_lower_score = 0
        best_higher_score = 0
        for lower_bound in np.arange(0.05, 0.3, 0.001):
            true_zero = gt[:, i][pred[:, i] <= lower_bound] == 0
            if len(true_zero) < 1:
                continue

            score = sum(true_zero) / len(true_zero)
            if score > best_lower_score and len(true_zero) > min_amount:
                lower = lower_bound
                best_lower_score = score

        for upper_bound in np.arange(0.8, 0.95, 0.001):
            true_one = gt[:, i][pred[:, i] >= upper_bound] == 1
            if len(true_one) < 1:
                continue

            score = sum(true_one) / len(true_one)
            if score > best_higher_score and len(true_one) > min_amount:
                upper = upper_bound
                best_higher_score = score

        threshold[i] = {
            'lower_bound': lower,
            'lower_accuracy': best_lower_score,
            'upper_bound': upper,
            'upper_accuracy': best_higher_score
        }
    return threshold

def predict_image_score(l, image_list, valid, batch_size=64, is_new=False):
    """
    This function stores the image names and its corresponding scores in parameter l.
    The is_new here means old or new model, will be replaced later.
    """
    if is_new:
        # net = network.ResNet()
        net = network.scalenet101(structure_path='structures/scalenet101.json')
        model_path = 'modelstates/big_scalenet101_last.pth'    # TODO: avoid hard code
        pretrained = torch.load(model_path)['model']
        pretrained_modify = {k[7:]: v for k, v in pretrained.items()}
        net.load_state_dict(pretrained_modify)
        # net.load_state_dict(pretrained)
    else:
        model_path = 'modelstates/model_last.pth'
        model_param = torch.load(model_path)['model']
        net = network.ResNet()
        net.load_state_dict(model_param)
    print(f'Model loaded from {model_path}')
    net.cuda()
    net.eval()

    image_batches = chunks(image_list, -1, batch_size)
    for image_batch in tqdm(image_batches):
        img_list = []
        for i in range(len(image_batch)):
            sub_image = Image.open(os.path.join(valid, image_batch[i]))
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            image = transform(sub_image)
            img_list.append(image)
        with torch.no_grad():
            image = torch.stack(img_list, dim=0).cuda()
            score = torch.sigmoid(net(image))
            score = score.cpu().numpy().reshape(len(image_batch), 3)
        l.extend(list(zip(image_batch, score)))


def chunks(lst, num_workers, n):
    """
    This function splits list of files into `chunk_list` for multiprocessing. n=-1 means the step is not specified
    and will be calculated in run time.
    """
    chunk_list = []
    if n == -1:
        n = ceil(len(lst)/num_workers)
        for i in range(0, len(lst), n):
            chunk_list.append(lst[i:i + n])
        return chunk_list
    else:
        for i in range(0, len(lst), n):
            chunk_list.append(lst[i:i + n])
        return chunk_list


def get_crop_label(score_path, threshold, save_folder):
    scores = np.load(score_path, allow_pickle=True)
    big_labels = []
    pred = []
    for i in range(len(scores)):
        pred.append(scores[i][1])
        label = scores[i][0][-13:-4]
        big_labels.append([int(label[1]), int(label[4]), int(label[7])])
    pred = np.stack(pred)
    big_labels = np.stack(big_labels)
    for i in range(3):
        pred[:, i][pred[:, i] <= threshold[str(i)]['lower_bound']] = 0
        pred[:, i][pred[:, i] >= threshold[str(i)]['upper_bound']] = 1
        pred[:, i][np.logical_and(
            pred[:, i] > threshold[str(i)]['lower_bound'], pred[:, i] < threshold[str(i)]['upper_bound'])] = -1
    pred = pred * big_labels
    indicies = np.where(np.all(pred != -1, axis=1))
    image_name = scores[indicies][:, 0]
    image_label = pred[indicies, :].astype(np.int8).reshape(-1, 3)
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    for i in tqdm(range(len(image_name))):
        image = Image.open(os.path.join(cut_result_path, image_name[i]))
        image.save(f'{save_folder}/{image_name[i][:-13]}{image_label[i]}.png')


def crop_train_set():
    dataset_path = 'Dataset/1.training'
    cut_result_path = "./train_single_patches"
    if not os.path.exists(cut_result_path):
        os.mkdir(cut_result_path)
    
    file_list = []
    for file in os.listdir(dataset_path):
        label = file.split('-')[-1][:-4]
        labels = [int(label[1]), int(label[4]), int(label[7])]
        if sum(labels) > 1:
            file_list.append((os.path.join(dataset_path, file), file[:-14], white_threshold, labels, cut_result_path, patch_shape, stride))
    process_map(crop_train_image, file_list, max_workers=6)

    save_score_name = 'patch9632_train'
    generate_image_label_score(cut_result_path, save_score_name, num_workers=1, batch_size=64, is_new=True)
    with open('prediction_threshold96.json') as json_file:
        prediction_threshold = json.load(json_file)
    
    get_crop_label(f'image_label_score/{save_score_name}.npy', prediction_threshold, 'patch9632_train')

def crop_valid_set():
    valid_mask_path = 'Dataset/2.validation/mask'
    valid_origin_path = 'Dataset/2.validation/img'
    cut_result_path = "./valid_single_patches"
    if not os.path.exists(cut_result_path):
        os.mkdir(cut_result_path)
    image_names = os.listdir(valid_mask_path)
    for image in tqdm(image_names):
        origin_image_path = os.path.join(valid_origin_path, image)
        mask_image_path = os.path.join(valid_mask_path, image)
        origin_im = np.asarray(Image.open(origin_image_path))
        mask_im = np.asarray(Image.open(mask_image_path))
        index = image[:2]
        crop_valid_image(origin_im, mask_im, index, threshold, white_threshold, cut_result_path)

def crop_train_set():
    dataset_path = 'Dataset/3.testing'
    cut_result_path = "./test_single_patches"
    pass

def crop_test():
    save_name = 'patch9632'
    valid = 'valid_single_patches/'
    generate_image_label_score(valid, save_name, num_workers=1, batch_size=64, is_new=True)
    prediction_threshold = test_crop_accuracy(f'image_label_score/{save_name}.npy', './val_labels.npy', min_amount=100)
    print(json.dumps(prediction_threshold, indent=4))
    with open('prediction_threshold96.json', 'w') as fp:
        json.dump(prediction_threshold, fp)