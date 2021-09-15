import os
os.environ['CUDA_VISIBLE_DEVICES']='1, 2, 3'
import torch
import network
import dataset
from torch.utils.data import DataLoader
import numpy as np
from torchvision import transforms
from tqdm import tqdm

def convertinttoonehot(nums_list):
    dic = {0: [1, 0, 0], 1: [0, 1, 0], 2: [0, 0, 1]}
    result = np.empty((len(nums_list), 3))
    for i in range(len(nums_list)):
        result[i] = np.array(dic[nums_list[i].item()])

    return torch.tensor(result, requires_grad=False)

if __name__ == "__main__":
    Dataset = dataset.SingleLabelDataset("valid_single_patches", transform=transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))
    
    save_path = './model.pth'
    state_dict = torch.load(save_path)
    net = network.ResNet().cuda()
    net.load_state_dict(state_dict)
    print(f'Model loaded from {save_path}')

    valid_loader = DataLoader(Dataset, batch_size=len(Dataset), num_workers=1, drop_last=True)
    correct = 0

    with torch.no_grad():
        net.eval()
        running_corrects = 0
        for inputs, labels in tqdm(valid_loader):
            inputs = inputs.cuda()
            onehot_label = convertinttoonehot(labels).cuda()

            outputs = net(inputs)

            _, preds = torch.argmax(outputs.data, dim=1)
            correct += torch.sum(preds == labels.data)
    
    print("accuracy: ", correct / len(Dataset))