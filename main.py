import dataset
import network
import train
from utils import crop_subpatches
import validation
import torch

if __name__ == '__main__':
    # step 1: crop valid images to small patches
    side_length = 96
    stride = 32
    white_threshold = 600
    cell_percentage = 0.1
    crop_subpatches.crop_valid_set(side_length, stride, white_threshold, cell_percentage)
    
    # step 2: use big label network predict the crops, get the best threshold
    save_name = 'big_label_9632'
    
    net = network.scalenet101(structure_path='structures/scalenet101.json')
    model_path = 'modelstates/big_scalenet101_last.pth'
    pretrained = torch.load(model_path)['model']
    pretrained_modify = {k[7:]: v for k, v in pretrained.items()}
    net.load_state_dict(pretrained_modify)
    
    crop_subpatches.valid_crop_test(save_name, net)
    
    # step 3: crop train images
    #   step 3.1: crop single label images
    #   step 3.2: crop mixed label images
    crop_subpatches.crop_train_set(white_threshold, side_length, stride)
    
    # step 4: use big label network predict the mixed-label image small crops under the threshold
    crop_subpatches.predict_and_save_train_crops(net, save_name)

    # step 4.5 balancing the train data
    

    # step 5: train small_crop network
    pass
