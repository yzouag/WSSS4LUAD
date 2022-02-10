import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2 as cv

def plot_uncertainty_map():
    pick_ckpts = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44]

    for im in os.listdir('wsss_valid_out_cam/multickpt'):
        binary_prediction_list = []
        for ckpt_folder in pick_ckpts:
            ckpt_folder_name = f'wsss_valid_out_cam/multickpt_{ckpt_folder}'
            cam = np.load(os.path.join(ckpt_folder_name, im), allow_pickle=True)
            # for i in range(3):
            #     cam[i] = (cam[i] - np.min(cam[i])) / (np.max(cam[i]) - np.min(cam[i]))
            binary_prediction_list.append(cam)

        binary_prediction_list = np.stack(binary_prediction_list, axis=0)
        for i in range(binary_prediction_list.shape[0]):
            binary_prediction_list[i] = (binary_prediction_list[i] - np.min(binary_prediction_list, axis=0)) / \
            (np.max(binary_prediction_list, axis=0) - np.min(binary_prediction_list, axis=0))
        variance = np.var(binary_prediction_list, axis=0)
        max_var = np.nanmax(variance, axis=0)
        max_var = (max_var - np.min(max_var)) / (np.max(max_var) - np.min(max_var)) # norm variance to [0, 1]
        
        
        plt.figure()
        plt.subplot(141)
        pred = np.load('wsss_valid_out_cam/multickpt/' + im, allow_pickle=True)
        im_name = im.split('.')[0] + '.png'
        gt = np.asarray(Image.open('Dataset_wsss/2.validation/mask/' + im_name))
        res = np.zeros([gt.shape[0], gt.shape[1], 3]).astype(np.uint8)
        res[gt == 0] = [0, 64, 128]
        res[gt == 1] = [64, 128, 0]
        res[gt == 2] = [243, 152, 0]
        res[gt == 3] = [255, 255, 255]
        cmap = plt.get_cmap('jet')
        rgb_img = cmap(max_var)[:, :, :3]
        rgb_img[gt == 3] = [0, 0, 0]
        plt.imshow(rgb_img)


        diff = np.logical_and((pred != gt), (gt != 3))
        rgb_img[~diff] = [0, 0, 0]
        plt.subplot(142)
        plt.imshow(rgb_img)
        # alpha = 0.5
        # dst = cv.addWeighted(rgb_img.astype(np.uint8), alpha, res.astype(np.uint8), 1-alpha, 0.0)
        # uncertainty = max_var > 0
        # res = np.zeros([diff.shape[0], diff.shape[1], 3]).astype(np.uint8)
        # res[diff] = [255, 0, 0]
        # res[uncertainty] = [0, 255, 0]
        # res[np.logical_and(diff, uncertainty)] = [0, 0, 255]
        # plt.imshow(dst)
        # plt.title(f'uncertain and high variance {np.sum(np.logical_and(diff, uncertainty))/np.sum(diff)}')
        
        
        plt.subplot(143)
        plt.imshow(res)
        
        
        plt.subplot(144)
        res = np.zeros([gt.shape[0], gt.shape[1], 3]).astype(np.uint8)
        pred[gt==3] = 3
        res[pred==0] = [0, 64, 128]
        res[pred==1] = [64, 128, 0]
        res[pred==2] = [243, 152, 0]
        res[pred==3] = [255, 255, 255]
        plt.imshow(res)
        plt.savefig(im_name)
        plt.close()
        np.save(f'max_var/{im}', max_var)

        for im in os.listdir('wsss_valid_out_cam/multickpt'):
            binary_prediction_list = []
            for ckpt_folder in pick_ckpts:
                ckpt_folder_name = f'wsss_valid_out_cam/multickpt_{ckpt_folder}'
                cam = np.load(os.path.join(ckpt_folder_name, im), allow_pickle=True)
                result_label = cam.argmax(axis=0)
                binary_mask = np.zeros_like(cam, dtype=np.int32)
                binary_mask[0, ...] = result_label == 0
                binary_mask[1, ...] = result_label == 1
                binary_mask[2, ...] = result_label == 2
                binary_prediction_list.append(binary_mask)

            binary_prediction_list = np.stack(binary_prediction_list, axis=0)
            variance = np.var(binary_prediction_list, axis=0)
            max_var = np.amax(variance, axis=0)
            # max_var = (max_var - np.min(max_var)) / (np.max(max_var) - np.min(max_var)) # norm variance to [0, 1]
            
            plt.figure()
            plt.subplot(131)
            pred = np.load('wsss_valid_out_cam/multickpt/' + im, allow_pickle=True)
            im_name = im.split('.')[0] + '.png'
            gt = np.asarray(Image.open('Dataset_wsss/2.validation/mask/' + im_name))
            
            diff = np.logical_and((pred != gt), (gt != 3))
            uncertainty = max_var > 0
            res_map = np.zeros([diff.shape[0], diff.shape[1], 3]).astype(np.uint8)
            res_map[diff] = [255, 0, 0]
            res_map[uncertainty] = [0, 255, 0]
            res_map[np.logical_and(diff, uncertainty)] = [0, 0, 255]
            with open('log.txt', 'a') as f:
                f.writelines(f'{im}: blue/blue+red {np.sum(np.logical_and(diff, uncertainty))/np.sum(diff)}, \
                green/correct {np.sum(np.all(res_map == [0,255,0], axis=-1))/(np.sum(~diff))}\n')
            plt.imshow(res_map)
            
            plt.subplot(132)
            res = np.zeros([gt.shape[0], gt.shape[1], 3]).astype(np.uint8)
            res[gt == 0] = [0, 64, 128]
            res[gt == 1] = [64, 128, 0]
            res[gt == 2] = [243, 152, 0]
            res[gt == 3] = [255, 255, 255]
            plt.imshow(res)
            
            
            plt.subplot(133)
            res = np.zeros([gt.shape[0], gt.shape[1], 3]).astype(np.uint8)
            pred[gt==3] = 3
            res[pred==0] = [0, 64, 128]
            res[pred==1] = [64, 128, 0]
            res[pred==2] = [243, 152, 0]
            res[pred==3] = [255, 255, 255]
            plt.imshow(res)
            plt.savefig(im_name)
            plt.close()

if __name__ == '__main__':
    pick_ckpts = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44]

    if not os.path.exists('train_pseudo_uncertain'):
        os.mkdir('train_pseudo_uncertain')

    for im in tqdm(os.listdir('Dataset_wsss/1.training/img/')):
        im_name = im.split('.')[0]
        binary_prediction_list = []
        for ckpt_folder in pick_ckpts:
            ckpt_folder_name = f'resnet_multickpt_{ckpt_folder}_train_pseudo_mask'
            result_label = np.load(os.path.join(ckpt_folder_name, f'{im_name}.npy'), allow_pickle=True)
            binary_mask = np.zeros((3, result_label.shape[0], result_label.shape[1]), dtype=np.int32)
            binary_mask[0, ...] = result_label == 1
            binary_mask[1, ...] = result_label == 2
            binary_mask[2, ...] = result_label == 3
            binary_prediction_list.append(binary_mask)

        binary_prediction_list = np.stack(binary_prediction_list, axis=0)
        variance = np.var(binary_prediction_list, axis=0)
        max_var = np.amax(variance, axis=0)
        max_var = (max_var - np.min(max_var)) / (np.max(max_var) - np.min(max_var)) * 255 # norm variance to [0, 1]

        result_label -= 1
        result_label[result_label == -1] = 255

        res = np.stack([result_label, max_var, max_var], axis=-1).astype(np.uint8)
        res = Image.fromarray(res).save(f'train_pseudo_uncertain/{im_name}.png')
    