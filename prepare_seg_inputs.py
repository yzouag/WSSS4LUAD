
import argparse
import os
from PIL import Image
import numpy as np
from scipy.stats.stats import mode
import torch
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from tqdm import tqdm
import torch.nn.functional as F
from dataset import TrainingSetCAM
import network
from utils.util import predict_mask

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-batch", default=20, type=int)
    parser.add_argument("-epoch", default=36, type=int)
    parser.add_argument("-lr", default=0.01, type=float)
    parser.add_argument("-resize", default=224, type=int)
    parser.add_argument("-side_length", default=224, type=int)
    parser.add_argument("-stride", default=int(224//3), type=int)
    parser.add_argument('-d','--device', nargs='+', help='GPU id to use parallel', required=True, type=int)
    parser.add_argument('-ckpt', type=str, help='the checkpoint model name')
    parser.add_argument("-c", default=2, type=int, help="number of classes")
    args = parser.parse_args()

    batch_size = args.batch
    epochs = args.epoch
    base_lr = args.lr
    resize = args.resize
    devices = args.device
    ckpt = args.ckpt
    side_length = args.side_length
    stride = args.stride
    num_class = args.c

    train_pseudo_mask_path = 'train_pseudo_mask'
    if not os.path.exists(train_pseudo_mask_path):
        os.mkdir(train_pseudo_mask_path)

    train_dataset_path = 'Dataset_crag/1.training/origin_ims'
    scales = [1, 1.25, 1.5, 1.75, 2]
    majority_vote = False
    
    # IMPORTANT! Modify the nprmalization part here!!
    dataset = TrainingSetCAM(data_path_name=train_dataset_path, transform=transforms.Compose([
                        transforms.Resize((resize,resize)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.678,0.505,0.735], std=[0.144,0.208,0.174])
                ]), patch_size=side_length, stride=stride, scales=scales, num_class=0
    )
    dataLoader = DataLoader(dataset, batch_size=1, drop_last=False)

    net_cam = network.wideResNet_cam(num_class=num_class)
    model_path = "modelstates/" + ckpt + ".pth"
    pretrained = torch.load(model_path)['model']
    pretrained = {k[7:]: v for k, v in pretrained.items()}
    pretrained['fc1.weight'] = pretrained['fc1.weight'].unsqueeze(-1).unsqueeze(-1).to(torch.float64)
    net_cam.load_state_dict(pretrained)

    net_cam.eval()
    net_cam = torch.nn.DataParallel(net_cam, device_ids=devices).cuda()

    with torch.no_grad():
        for im_name, scaled_im_list, scaled_position_list, scales, big_label in tqdm(dataLoader):
            big_label = big_label[0]
            eliminate_noise = True
            if big_label.item()==0:
                eliminate_noise = False
            # exit()
            orig_img = np.asarray(Image.open(f'{train_dataset_path}/{im_name[0]}'))
            w, h, _ = orig_img.shape

            if majority_vote:
                ensemble_cam = []
            else:
                ensemble_cam = np.zeros((num_class, w, h))

            # get the prediction for each pixel in each scale
            for s in range(len(scales)):
                w_ = int(w*scales[s])
                h_ = int(h*scales[s])
                interpolatex = side_length
                interpolatey = side_length

                if w_ < side_length:
                    interpolatex = w_
                if h_ < side_length:
                    interpolatey = h_

                im_list = scaled_im_list[s]
                position_list = scaled_position_list[s]

                im_list = torch.vstack(im_list)
                im_list = torch.split(im_list, batch_size)

                cam_list = []
                for ims in im_list:
                    cam_scores = net_cam(ims.cuda())
                    cam_scores = F.interpolate(cam_scores, (interpolatex, interpolatey), mode='bilinear', align_corners=False).detach().cpu().numpy()
                    cam_list.append(cam_scores)
                cam_list = np.concatenate(cam_list)

                sum_cam = np.zeros((num_class, w_, h_))
                sum_counter = np.zeros_like(sum_cam)
            
                for k in range(cam_list.shape[0]):
                    y, x = position_list[k][0], position_list[k][1]
                    crop = cam_list[k]
                    sum_cam[:, y:y+side_length, x:x+side_length] += crop
                    sum_counter[:, y:y+side_length, x:x+side_length] += 1
                sum_counter[sum_counter < 1] = 1

                norm_cam = sum_cam / sum_counter
                norm_cam = F.interpolate(torch.unsqueeze(torch.tensor(norm_cam),0), (w, h), mode='bilinear', align_corners=False).detach().cpu().numpy()[0]
                # use the image-level label to eliminate impossible pixel classes
                if majority_vote:
                    if eliminate_noise:
                        for k in range(num_class):
                            if big_label[k] == 0:
                                norm_cam[k, :, :] = -np.inf
                
                    norm_cam = np.argmax(norm_cam, axis=0)        
                    ensemble_cam.append(norm_cam)
                else:
                    ensemble_cam += norm_cam
            
            if majority_vote:
                ensemble_cam = np.stack(ensemble_cam, axis=0)
                result_label = mode(ensemble_cam, axis=0)[0]
            else:
                if eliminate_noise:
                    for k in range(num_class):
                        if big_label[k] == 0:
                            ensemble_cam[k, :, :] = -np.inf
                            
                result_label = ensemble_cam.argmax(axis=0)
            
            result_label = result_label + 1
            predicted_background_mask = predict_mask(Image.open(f'{train_dataset_path}/{im_name[0]}'), 230, 50)
            result_label = predicted_background_mask * result_label

            if not os.path.exists(f'{train_pseudo_mask_path}'):
                os.mkdir(f'{train_pseudo_mask_path}')
            np.save(f'{train_pseudo_mask_path}/{im_name[0][:-4]}.npy', result_label)
