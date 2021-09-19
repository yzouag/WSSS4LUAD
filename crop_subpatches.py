from multiprocessing.context import Process
import os
import numpy as np
from PIL import Image
from patchify import patchify
import argparse
from multiprocessing import Manager, Pool
import shutil
from tqdm import tqdm
import torch
import network
from torchvision import transforms
from math import ceil
from sklearn.metrics import f1_score

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


# # todo: add consideration for the whole image, need to run paralelly and add a gridsearch for threshold
# # change the output to sigmoid
# def get_label_from_nn(sub_image, f, upper=1, lower=-1):
#     im_type = [0, 0, 0]
#     transform = transforms.Compose([
#         transforms.ToPILImage(),
#         transforms.Resize(224),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
#     image = transform(sub_image)
#     image = image.reshape((1, 3, 224, 224))
#     with torch.no_grad():
#         input = image.cuda()
#         score = torch.sigmoid(net(input))
#         result = score.cpu().numpy().reshape(-1)
#     f.write(str(result) + '\n')
#     for i in range(3):
#         if result[i] > upper:
#             im_type[i] = 1
#         elif result[i] < lower:
#             im_type[i] = 0
#         else:
#             im_type[i] = -1
#     return im_type


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


def generate_image_label_score(test_path, save_name, num_workers=3, batch_size=64):
    files = os.listdir(test_path)
    image_chunks = chunks(files, num_workers, -1)
    
    with Manager() as manager:
        L = manager.list()
        processes = []
        for i in range(num_workers):
            p = Process(target=predict_image_score, args=(L,image_chunks[i], test_path, batch_size))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        if not os.path.exists('image_label_score'):
            os.mkdir('image_label_score')
        np.save(f'image_label_score/{save_name}.npy', list(L))

def test_crop_accuracy(score_path):
    scores = np.load(score_path, allow_pickle=True)
    gt = []
    pred = []
    for i in range(len(scores)):
        pred.append(scores[i][1])
        label = scores[i][0][-13:-4]
        gt.append([int(label[1]), int(label[4]), int(label[7])])
    pred = np.stack(pred)
    gt = np.stack(gt)

    best_f1_score = 0
    for lower_bound in np.arange(0,0.5,0.02):
        for upper_bound in np.arange(0.5,1,0.02):
            for i in range(3):
                pred[:,i][pred[:,i] < lower_bound] = 0
                pred[:,i][pred[:,i] > upper_bound] = 1
                pred[:,i][np.logical_and(pred[:,i] >= lower_bound,pred[:,i] <= upper_bound)] = -1
            print(pred.transpose().shape)
            print(gt.shape)
            score = f1_score(gt.transpose(), pred.transpose(), average='macro')
            if score > best_f1_score:
                best_f1_score = score
                threshold = (lower_bound, upper_bound)
    return best_f1_score, threshold
            

def make_chunk(target_list, n):
    for i in range(0, len(target_list), n):
        yield target_list[i:i + n]

def predict_image_score(l, image_list, valid, batch_size=64):
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
            sub_image = np.asarray(Image.open(valid + image_batch[i]))
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(224),
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

    if args.test:
        # valid = 'valid_single_patches/'
        # generate_image_label_score(valid, 'validation', num_workers=1)
        print(test_crop_accuracy('image_label_score/validation.npy'))
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
