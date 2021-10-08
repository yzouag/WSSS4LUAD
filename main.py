import argparse
import dataset
import network
from train import train_integrated
from utils import crop_subpatches, generate_CAM, visualization
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", default=64, type=int)
    parser.add_argument("-epoch", default=30, type=int)
    parser.add_argument("-lr", default=0.001, type=float)
    parser.add_argument("-resize", default=448, type=int)
    parser.add_argument("-step", default=10, type=int)
    parser.add_argument("-save_every", default=10, type=int, help="how often to save a model while training")
    parser.add_argument("-gamma", default=0.1, type=float)
    parser.add_argument('-d','--device', nargs='+', help='GPU id to use parallel', required=True, type=int)
    parser.add_argument('-m', type=str, help='the save model name', required=True)
    parser.add_argument('-crop', action='store_true', help='if execute crop phase')
    parser.add_argument('-side_length', default=96, type=int, help='crop size')
    parser.add_argument('-stride', type=int, default=32, help='crop step size')
    parser.add_argument('-white_threshold', type=int, default=600, help='the threshold for asserting the pixel is white')
    parser.add_argument('-cell_percent', type=float, default=0.1, help='the threshold for asserting one image contain certain label cells')
    parser.add_argument('-threshold_file_name', type=str, default='prediction_threshold', help='the name of the threshold file')
    args = parser.parse_args()

    batch_size = args.batch
    epochs = args.epoch
    base_lr = args.lr
    resize = args.resize
    step_size = args.step
    save_every = args.save_every
    gamma = args.gamma
    devices = args.device
    setting_str = args.m
    if_crop = args.crop
    side_length = args.side_length
    stride = args.stride
    white_threshold = args.white_threshold
    cell_percentage = args.cell_percent
    threshold_file_name = args.threshold_file_name

    if if_crop:
        # step 1: crop valid images to small patches
        crop_subpatches.crop_valid_set(side_length, stride, white_threshold, cell_percentage)

        # step 2: use big label network predict the crops, get the best threshold
        net = network.scalenet101(structure_path='structures/scalenet101.json')
        model_path = 'modelstates/big_scalenet101_last.pth'
        pretrained = torch.load(model_path)['model']
        pretrained_modify = {k[7:]: v for k, v in pretrained.items()}
        net.load_state_dict(pretrained_modify)

        crop_subpatches.valid_crop_test(threshold_file_name, net)

        # step 3: crop train images
        #   step 3.1: crop single label images
        #   step 3.2: crop mixed label images
        crop_subpatches.crop_train_set(white_threshold, side_length, stride)

        # step 4: use big label network predict the mixed-label image small crops under the threshold
        crop_subpatches.predict_and_save_train_crops(net, threshold_file_name)

    # step 4.5 balancing the train data

    # step 5: train small_crop network
    net = network.scalenet101(structure_path='../structures/scalenet101.json', ckpt='../weights/scalenet101.pth')
    net = torch.nn.DataParallel(net, device_ids=devices).cuda()
    transform = transforms.Compose([
        transforms.Resize((resize, resize)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    TrainDataset = dataset.DoubleLabelDataset(transform=transform)
    print("train Dataset", len(TrainDataset))
    TrainDatasampler = torch.utils.data.RandomSampler(TrainDataset)
    TrainDataloader = DataLoader(TrainDataset, batch_size=batch_size, num_workers=2, sampler=TrainDatasampler, drop_last=True)
    
    ValidDataset = dataset.ValidationDataset(transform=transform)
    print("valid Dataset", len(ValidDataset))
    ValidDataloader = DataLoader(ValidDataset, batch_size=batch_size, num_workers=2, drop_last=True)

    optimizer = torch.optim.SGD(net.parameters(), base_lr, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    criteria = torch.nn.BCEWithLogitsLoss(reduction='mean')
    criteria.cuda()

    train_integrated.train_small_label(net, TrainDataloader, ValidDataloader, optimizer, criteria, scheduler, epochs, save_every, setting_str)
    
    # step 6: generate CAM
    dataset_path = './Dataset/2.validation/img'
    model_name = 'secondphase_scalenet101_224_last'
    model_crop = (96, 32)
    generate_CAM.generate_cam(net, dataset_path, model_name, model_crop, batch_size, mode='valid')

    # step 6.5: generate visualization result and validation
    visualization.visualize_result(f'out_cam/{model_name}')

    dataset_path = './Dataset/3.testing/img'
    generate_CAM.generate_cam(net, dataset_path, model_name, model_crop, batch_size, mode='train')

    # step 7: train segmentation network

    # step 8: make segmentation prediction

    # step 9: post processing
    pass
