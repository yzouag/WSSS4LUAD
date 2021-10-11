import argparse
import dataset
import network
from utils import crop_subpatches, generate_CAM, visualization
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4,6'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-side_length', default=96, type=int, help='crop size')
    parser.add_argument('-stride', type=int, default=32, help='crop step size')
    parser.add_argument('-white_threshold', type=int, default=0.9, help='the threshold for asserting the pixel is white')
    parser.add_argument('-cell_percent', type=float, default=0.1, help='the threshold for asserting one image contain certain label cells')
    parser.add_argument('-threshold_file_name', type=str, default='prediction_threshold', help='the name of the threshold file')
    args = parser.parse_args()

    side_length = args.side_length
    stride = args.stride
    white_threshold = args.white_threshold
    cell_percentage = args.cell_percent
    threshold_file_name = args.threshold_file_name

    # step 1: crop valid images to small patches
    print('crop valid images to small patches...')
    crop_subpatches.crop_valid_set(side_length, stride, white_threshold, cell_percentage)

    # step 2: use big label network predict the crops, get the best threshold
    net = network.scalenet101(structure_path='structures/scalenet101.json')
    model_path = 'modelstates/big_scalenet101_last.pth'
    pretrained = torch.load(model_path)['model']
    pretrained_modify = {k[7:]: v for k, v in pretrained.items()}
    net.load_state_dict(pretrained_modify)

    print('valid crop and get the threshold...')
    crop_subpatches.valid_crop_test(threshold_file_name, net)

    # step 3: crop train images
    #   step 3.1: crop single label images
    #   step 3.2: crop mixed label images
    print('crop train images')
    crop_subpatches.crop_train_set(white_threshold, side_length, stride)

    # step 4: use big label network predict the mixed-label image small crops under the threshold
    print('predict train image labels')
    crop_subpatches.predict_and_save_train_crops(net, threshold_file_name)

    
    # step 6: generate CAM
    # dataset_path = './Dataset/2.validation/img'
    # model_name = 'secondphase_scalenet101_224_last'
    # model_crop = (96, 32)
    # generate_CAM.generate_cam(net, dataset_path, model_name, model_crop, batch_size, mode='valid')

    # step 6.5: generate visualization result and validation
    # visualization.visualize_result(f'out_cam/{model_name}')

    # dataset_path = './Dataset/3.testing/img'
    # generate_CAM.generate_cam(net, dataset_path, model_name, model_crop, batch_size, mode='train')

    # step 7: train segmentation network

    # step 8: make segmentation prediction

    # step 9: post processing