import json
from torchvision import transforms
from tqdm import tqdm
import numpy as np
from PIL import Image
import os
import network
import torch

def cut_patches(im, im_size, stride, transform):
    im_list = []
    h, w, _ = im.shape

    h_ = np.arange(0, h - im_size + 1, stride)
    if h % stride != 0:
        h_ = np.append(h_, h-im_size)
    w_ = np.arange(0, w - im_size + 1, stride)
    if w % stride != 0:
        w_ = np.append(w_, w - im_size)

    for i in h_:
        for j in w_:   	
            temp = Image.fromarray(np.uint8(im[i:i+im_size,j:j+im_size,:]))
            temp = transform(temp)
            im_list.append(temp)
    return torch.stack(im_list)


def predict_big_label(image_path, im_size, stride, threshold, model_path):
    net = network.ResNet()
    pretrained = torch.load(model_path)['model']
    pretrained_modify = {k[7:]: v for k, v in pretrained.items()}
    net.load_state_dict(pretrained_modify)
    net.cuda()
    net.eval()

    image_names = os.listdir(image_path)
    image_labels = []
    for image in tqdm(image_names):
        im = np.asarray(Image.open(os.path.join(image_path, image)))
        transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        im_list = cut_patches(im, im_size=im_size, stride=stride, transform=transform).cuda()
        label = [0, 0, 0]
        total_scores = []
        for ims in torch.split(im_list, 16):
            with torch.no_grad():
                scores = torch.sigmoid(net(ims))
                total_scores.append(scores.cpu().numpy())
        scores = np.vstack(total_scores)
        for i in range(3):
            if sum(scores[: ,i] > threshold[i]) >= 2:
                label[i] = 1
        image_labels.append(label)
    return image_labels

threshold = [0.83, 0.21, 0.96]
validation = predict_big_label('Dataset/2.validation/img', 225, 80, threshold, 'modelstates/01_best.pth')
images = os.listdir('/Dataset/2.validation/mask')
gt = []
for image in images:
    image_path = os.path.join('/Dataset/2.validation/mask', image)
    im = np.asarray(Image.open(image_path))
    label = [0,0,0]
    im_label = np.unique(im)
    for i in range(3):
        if i in im_label:
            label[i] = 1
    gt.append(label)

res = {
    'pred': validation,
    'groundtruth': gt
}
with open('result.json', 'w') as fp:
    json.dump(res, fp)

count = 0
for i in range(len(validation)):
    if validation[i] == gt[i]:
        count += 1
print(f'accuracy: {count/len(validation)}')