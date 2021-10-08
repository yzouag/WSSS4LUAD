import os
import numpy as np
from PIL import Image
from utils.util import online_cut_patches
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import torch
from torchvision import transforms
from math import ceil
import json
import shutil


def crop_train_image(file_info):
    imfile, count, threshold, labels, cut_result_path, patch_shape, stride = file_info
    im = Image.open(imfile)
    im_arr = np.asarray(im)

    # exclude images that are smaller than the crop size
    if im_arr.shape[0] < patch_shape or im_arr.shape[1] < patch_shape:
        return

    patches = online_cut_patches(im_arr, patch_shape, stride)
    
    for i in range(len(patches)):
        sub_image = patches[i]
        if is_valid_crop(sub_image, threshold, groundtruth=False):
            result = Image.fromarray(np.uint8(sub_image))
            result.save(os.path.join(cut_result_path, str(count) + "_" + str(i) + str(labels) + '.png'))




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


def get_labels(label, cell_percentage):
    """
    This function generates the one-hot label for the given 2D groundtruth by the given activation threshold.
    """
    pix_type, pix_count = np.unique(label, return_counts=True)
    im_type = [0, 0, 0, 0]
    for i in range(len(pix_type)):
        if pix_count[i] / label.size > cell_percentage:
            im_type[pix_type[i]] = 1
    return im_type[:3]


def generate_image_label_score(test_path, net, batch_size=64):
    """
    use big_label network to predict the labels for each small images

    Args:
        test_path (str): the path of images
        net (network): the network for prediction
        batch_size (int, optional): Defaults to 64.

    Returns:
        list: a list of tuple (image_name, image_score)
    """
    files = os.listdir(test_path)
    scores_list = []
    
    net.cuda()
    net.eval()
    
    image_batches = chunks(files, -1, batch_size)
    for image_batch in tqdm(image_batches):
        img_list = []
        for i in range(len(image_batch)):
            sub_image = Image.open(os.path.join(test_path, image_batch[i]))
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
        scores_list.extend(list(zip(image_batch, score)))
    return scores_list


def test_crop_accuracy(score_list, min_amount=100):
    """
    use the predict scores to get the best threshold for each label
    notice for validation set, the number of crops with stroma as 1 is very low,
    so min amount cannot be too large.

    Args:
        score_list (list): a list of tuple (image_name, score_list)
        min_amount (int, optional): Default is 100. Some threshold is too high that it cannot get enough confident images.

    Returns:
        dict: threshold and their corresponding accuracies
    """
    gt = []
    pred = []
    for i in range(len(score_list)):
        pred.append(score_list[i][1])
        label = score_list[i][0][-13:-4]
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


def chunks(lst, num_workers=None, n=None):
    """
    a helper function for seperate the list to chunks

    Args:
        lst (list): the target list
        num_workers (int, optional): Default is None. When num_workers are not None, the function divide the list into num_workers chunks
        n (int, optional): Default is None. When the n is not None, the function divide the list into n length chunks

    Returns:
        llis: a list of small chunk lists
    """
    chunk_list = []
    if num_workers is None and n is None:
        print("the function should at least pass one positional argument")
        exit()
    elif n == None:
        n = ceil(len(lst)/num_workers)
        for i in range(0, len(lst), n):
            chunk_list.append(lst[i:i + n])
        return chunk_list
    else:
        for i in range(0, len(lst), n):
            chunk_list.append(lst[i:i + n])
        return chunk_list


def save_high_score_train_images(scores_list, threshold):
    """
    use the prediction scores and threhold to get high confidence images and save them back to folders

    Args:
        scores_list (list): list of tuples (image_name, image_scores)
        threshold (dict): the dictionary for different thresholds
    """
    cut_multiple_path = './train_multiple_label_patches'
    cut_temp_path = './train_multiple_label_patches_temp'

    pred = []
    big_labels = []
    for i in range(len(scores_list)):
        pred.append(scores_list[i][1])
        label = scores_list[i][0][-13:-4] # this will get a string '[0, 1, 1]'
        big_labels.append([int(label[1]), int(label[4]), int(label[7])])
    
    pred = np.stack(pred)
    big_labels = np.stack(big_labels)
    for i in range(3):
        pred[:, i][pred[:, i] <= threshold[str(i)]['lower_bound']] = 0
        pred[:, i][pred[:, i] >= threshold[str(i)]['upper_bound']] = 1
        pred[:, i][np.logical_and(pred[:, i] > threshold[str(i)]['lower_bound'], pred[:, i] < threshold[str(i)]['upper_bound'])] = -1
    
    pred = pred * big_labels
    indicies = np.where(np.all(pred != -1, axis=1))
    image_name = scores_list[indicies][:, 0]
    image_label = pred[indicies, :].astype(np.int8).reshape(-1, 3)
    
    if not os.path.exists(cut_multiple_path):
        os.mkdir(cut_multiple_path)
    
    for i in tqdm(range(len(image_name))):
        image = Image.open(os.path.join(cut_temp_path, image_name[i]))
        image.save(f'{cut_multiple_path}/{image_name[i][:-13]}{image_label[i]}.png')
    shutil.rmtree(cut_temp_path)
    print('generate multiple label data complete')
    print(f'number of images in multiple label: {len(os.listdir(cut_multiple_path))}')


def crop_train_set(white_threshold, side_length, stride):
    """
    crop the training set to small patches, the results are in two folders,
    `train_single_label_patches`, `train_multiple_label_patches`, notice they
    are extremely unbalanced. Now the patchify is deprecated.

    Args:
        white_threshold (int): rgb colors from 0~255, add up together
        side_length (int): the size of the image
        stride (int): step for each crop
    """
    dataset_path = 'Dataset/1.training'
    cut_single_path = './train_single_label_patches'
    cut_multiple_path = './train_multiple_label_patches_temp'

    # make the directory
    if not os.path.exists(cut_single_path):
        os.mkdir(cut_single_path)
    if not os.path.exists(cut_multiple_path):
        os.mkdir(cut_multiple_path)
    
    file_list = []
    for file in os.listdir(dataset_path):
        label = file.split('-')[-1][:-4]
        labels = [int(label[1]), int(label[4]), int(label[7])]
        if sum(labels) > 1:
            file_list.append((os.path.join(dataset_path, file), file[:-14], white_threshold, labels, cut_multiple_path, side_length, stride))
        else:
            file_list.append((os.path.join(dataset_path, file), file[:-14], white_threshold, labels, cut_single_path, side_length, stride))
    print('generate crop label images...')
    process_map(crop_train_image, file_list, max_workers=6, chunksize=10)

    print('cut complete')
    print('images for single label cut: ', len(os.listdir(cut_single_path)))
    print('images for multiple label cut: ', len(os.listdir(cut_multiple_path)))



def predict_and_save_train_crops(net, threshold_file_name):
    """
    predict image labels from `train_multiple_label_patches` and save high confidence images
    back to `train_multiple_label_patches`

    Args:
        net (network): the big patches label for predict image label
        threshold_file_name (str): the file name for threhold json file
    """
    test_path = './train_multiple_label_patches_temp'
    scores_list = generate_image_label_score(test_path, net, batch_size=64)
    with open(f'{threshold_file_name}.json') as json_file:
        prediction_threshold = json.load(json_file)
    save_high_score_train_images(scores_list, prediction_threshold)

def crop_valid_set(side_length, stride, white_threshold, cell_percentage):
    """
    crop valid images for testing, the images are stored in `valid_patches`

    Args:
        side_length (int): the patch's length
        stride (int): the length of the foot step
        white_threshold (int): sum of the rgb values, each 0~255
        cell_percentage (float): only cells make up x percents will be treated as 1
    """
    
    valid_mask_path = 'Dataset/2.validation/mask'
    valid_origin_path = 'Dataset/2.validation/img'
    cut_result_path = './valid_patches'
    
    if not os.path.exists(cut_result_path):
        os.mkdir(cut_result_path)
    
    image_names = os.listdir(valid_mask_path)

    for image in tqdm(image_names):
        origin_image_path = os.path.join(valid_origin_path, image)
        mask_image_path = os.path.join(valid_mask_path, image)
        origin_im = np.asarray(Image.open(origin_image_path))
        mask_im = np.asarray(Image.open(mask_image_path))
        
        index = image[:2]
        stack_image = np.concatenate((origin_im, mask_im.reshape(mask_im.shape[0], mask_im.shape[1], 1)), axis=2)
        patches = online_cut_patches(stack_image, side_length, stride)
        for i in range(len(patches)):
            sub_image = patches[i][:, :, :3]
            label = patches[i][:, :, 3]
            if is_valid_crop(label, white_threshold, groundtruth=True):
                im_type = get_labels(label, cell_percentage)
                result = Image.fromarray(np.uint8(sub_image))
                result.save(cut_result_path + '/image' + index + '_' + str(i) + '_' + str(im_type) + '.png')


def valid_crop_test(save_name, net):
    """
    apply the big_label network on validation small crops and get the best threshold, save it in a json file

    Args:
        save_name (str): the name of the final threshold json file
        net (network): the big label net for prediction
    """
    valid_crop_path = 'valid_patches/'
    score_list = generate_image_label_score(valid_crop_path, net, batch_size=64)
    prediction_threshold = test_crop_accuracy(score_list, min_amount=100)
    with open(f'{save_name}.json', 'w') as fp:
        json.dump(prediction_threshold, fp)