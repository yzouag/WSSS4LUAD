import argparse
from PIL import Image
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from torchvision import transforms
import network
import dataset
from torch.utils.data import DataLoader
from utils.metric import get_overall_valid_score
from utils.generate_CAM import generate_cam

# def train_small_label(net, train_loader, valid_side_length, optimizer, criteria, scheduler, epochs, save_every, setting_str, batch_size):
#     """
#     the training and validation function for small crop images

#     Args:
#         net (network): the model for training
#         train_loader (dataloader): train data loader
#         valid_side_length (int): the side length of the valid image crop
#         optimizer (optim): the optimizer for the training
#         criteria (loss): the loss function for training
#         scheduler (scheduler): training scheduler
#         epochs (int): number of training epochs
#         save_every (int): number of epochs to save one model
#         setting_str (str): the model name
#     """
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-batch", default=24, type=int)
    parser.add_argument("-epoch", default=15, type=int)
    parser.add_argument("-lr", default=0.001, type=float)
    parser.add_argument("-resize", default=448, type=int)
    parser.add_argument("-step", default=5, type=int, help='the step size for the scheduler')
    parser.add_argument("-save_every", default=10, type=int, help="how often to save a model while training")
    parser.add_argument("-gamma", default=0.1, type=float)
    parser.add_argument('-d','--device', nargs='+', help='GPU id to use parallel', required=True, type=int)
    parser.add_argument('-m', type=str, help='the save model name', required=True)
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

    net = network.scalenet101(structure_path='structures/scalenet101.json', ckpt='weights/scalenet101.pth')
    net = torch.nn.DataParallel(net, device_ids=devices).cuda()
    train_transform = transforms.Compose([
        transforms.Resize((resize, resize)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # val_transform = transforms.Compose([
    #     transforms.Resize((resize, resize)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])

    TrainDataset = dataset.DoubleLabelDataset(transform=train_transform)
    print("train Dataset", len(TrainDataset))
    TrainDatasampler = torch.utils.data.RandomSampler(TrainDataset)
    TrainDataloader = DataLoader(TrainDataset, batch_size=batch_size, num_workers=2, sampler=TrainDatasampler, drop_last=True)
    
    # ValidDataset = dataset.ValidationDataset("valid_patches", transform=val_transform)
    # print("valid Dataset", len(ValidDataset))
    # ValidDataloader = DataLoader(ValidDataset, batch_size=batch_size, num_workers=2, drop_last=True)

    optimizer = torch.optim.SGD(net.parameters(), base_lr, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    criteria = torch.nn.BCEWithLogitsLoss(reduction='mean')
    criteria.cuda()

    valid_image_path = 'valid_patches'
    valid_image_path = os.path.join(valid_image_path, os.listdir(valid_image_path)[0])
    side_length = np.asarray(Image.open(valid_image_path)).shape[0]
    print('side length: ', side_length)

    if not os.path.exists('modelstates'):
        os.mkdir('modelstates')

    loss_t = []
    accuracy_t = []
    iou_v = []
    best_val = 0

    for i in range(epochs):
        count = 0
        running_loss = 0.
        correct = 0
        net.train()

        for img, label in tqdm(TrainDataloader):
            count += 1
            img = img.cuda()
            label = label.cuda()
            scores = net(img)
            loss = criteria(scores, label.float())
            
            scores = torch.sigmoid(scores)
            predict = torch.zeros_like(scores)
            predict[scores > 0.5] = 1
            predict[scores < 0.5] = 0
            for k in range(len(predict)):
                if torch.equal(predict[k], label[k]):
                    correct += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / count
        train_acc = correct / (count * batch_size)
        scheduler.step()
        accuracy_t.append(train_loss)
        loss_t.append(train_acc)

        # # if i % 3 == 0:
        # with torch.no_grad():
        #     net.eval()
        #     running_loss = 0.0
        #     running_corrects = 0
        #     count = 0
        #     for img, label in tqdm(valid_loader):
        #         count += 1
        #         img = img.cuda()
        #         label = label.cuda()
        #         scores = net(img)
        #         loss = criteria(scores, label.float())
                
        #         scores = torch.sigmoid(scores)
        #         predict = torch.zeros_like(scores)
        #         predict[scores > 0.5] = 1
        #         predict[scores <= 0.5] = 0
        #         for k in range(len(predict)):
        #             if torch.equal(predict[k], label[k]):
        #                 running_corrects += 1
        #         running_loss += loss.item()
        #     valid_loss = running_loss / count
        #     valid_acc = running_corrects / (count * batch_size)
        #     loss_v.append(valid_loss)
        #     accuracy_v.append(valid_acc)
        net_cam = network.scalenet101_cam(structure_path='structures/scalenet101.json')
        pretrained = net.state_dict()
        pretrained = {k[7:]: v for k, v in pretrained.items()}
        pretrained['fc1.weight'] = pretrained['fc1.weight'].unsqueeze(-1).unsqueeze(-1).to(torch.float64)
        pretrained['fc2.weight'] = pretrained['fc2.weight'].unsqueeze(-1).unsqueeze(-1).to(torch.float64)
        net_cam.load_state_dict(pretrained)
        generate_cam(net_cam, setting_str, tuple((side_length, side_length//3)), batch_size, 'valid', resize)
        valid_image_path = f'valid_out_cam/{setting_str}'
        valid_iou = get_overall_valid_score(valid_image_path)
        iou_v.append(valid_iou)
        
        if valid_iou > best_val:
            print("Updating the best model..........................................")
            best_val = valid_iou
            torch.save({"model": net.state_dict(), 'optimizer': optimizer.state_dict()}, "./modelstates/" + setting_str + "_best.pth")
    
        print(f'Epoch [{i+1}/{epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Valid mIOU: {valid_iou:.4f}')

        if (i + 1) % save_every == 0 and (i + 1) != epochs:
            torch.save({"model": net.state_dict(), 'optimizer': optimizer.state_dict()}, "./modelstates/" + setting_str + "_ep"+str(i+1)+".pth")

    torch.save({"model": net.state_dict(), 'optimizer': optimizer.state_dict()}, "./modelstates/" + setting_str + "_last.pth")

    plt.figure(1)
    plt.plot(loss_t)
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.title('train loss')
    plt.savefig('./image/train_loss.png')
    plt.close()

    plt.figure(2)
    plt.plot(accuracy_t)
    plt.ylabel('accuracy')
    plt.xlabel('epochs')
    plt.title('train accuracy')
    plt.savefig('./image/train_accuracy.png')

    plt.figure(3)
    plt.plot(iou_v)
    plt.ylabel('accuracy')
    plt.xlabel('epochs')
    plt.title('valid accuracy')
    plt.savefig('./image/valid_iou.png')