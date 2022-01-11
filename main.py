import argparse
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from torchvision import transforms
import network
import dataset
from torch.utils.data import DataLoader
from utils.metric import get_overall_valid_score
from utils.generate_CAM import generate_validation_cam
from utils.mixup import Mixup
from timm.data.auto_augment import rand_augment_transform

class PolyOptimizer(torch.optim.SGD):
    def __init__(self, params, lr, weight_decay, max_step, momentum=0.9):
        super().__init__(params, lr, weight_decay)

        self.global_step = 0
        self.max_step = max_step
        self.momentum = momentum

        self.__initial_lr = [group['lr'] for group in self.param_groups]


    def step(self, closure=None):

        if self.global_step < self.max_step:
            lr_mult = (1 - self.global_step / self.max_step) ** self.momentum

            for i in range(len(self.param_groups)):
                self.param_groups[i]['lr'] = self.__initial_lr[i] * lr_mult

        super().step(closure)

        self.global_step += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-batch", default=20, type=int)
    parser.add_argument("-epoch", default=20, type=int)
    parser.add_argument("-lr", default=0.01, type=float)
    parser.add_argument("-resize", default=224, type=int)
    parser.add_argument("-save_every", default=0, type=int, help="how often to save a model while training")
    parser.add_argument("-test_every", default=5, type=int, help="how often to test a model while training")
    parser.add_argument("-c", default=3, type=int, help="number of classes")
    parser.add_argument('-d','--device', nargs='+', help='GPU id to use parallel', required=True, type=int)
    parser.add_argument('-m', type=str, help='the save model name')
    parser.add_argument('-resnet', action='store_true', default=False)
    parser.add_argument('-test', action='store_true', default=False)
    parser.add_argument('-ckpt', type=str, help='the checkpoint model name')
    parser.add_argument('-note', type=str, help='special experiments with this training', required=False)
    parser.add_argument("-cutmix", type=float, default="0.0", help="alpha value of beta distribution in cutmix, 0 to disable")
    parser.add_argument("-adl_threshold", type=float, default="0.0", help="range (0,1], the threhold for defining the salient activation values, 0 to disable")
    parser.add_argument("-adl_drop_rate", type=float, default="0.0", help="range (0,1], the possibility to drop the high activation areas, 0 to disable")
    parser.add_argument('-randaug', action='store_true', default=False)
    parser.add_argument("-reg", action='store_true', default=False, help="whether to use the area regression")
    parser.add_argument('-dataset', default='warwick', type=str, choices=['warwick', 'wsss'], help='now only support two types: (warwick, wsss)')
    args = parser.parse_args()

    batch_size = args.batch
    epochs = args.epoch
    base_lr = args.lr
    resize = args.resize
    num_class = args.c
    test_every = args.test_every
    devices = args.device
    model_name = args.m
    useresnet = args.resnet
    testonly = args.test
    ckpt = args.ckpt
    remark = args.note
    cutmix_alpha = args.cutmix
    adl_threshold = args.adl_threshold
    adl_drop_rate = args.adl_drop_rate
    rand_aug = args.randaug
    activate_regression = args.reg
    target_dataset = args.dataset
    if cutmix_alpha == 0:
        activate_regression = False

    if not os.path.exists('modelstates'):
        os.mkdir('modelstates')
    if not os.path.exists('val_image_label'):
        os.mkdir('val_image_label')
    if not os.path.exists('result'):
        os.mkdir('result')
    
    validation_cam_folder_name = f'{target_dataset}_valid_out_cam'
    validation_dataset_path = f'Dataset_{target_dataset}/2.validation/img'
    scales = [1, 1.25, 1.5, 1.75, 2]
    if not os.path.exists(validation_cam_folder_name):
        os.mkdir(validation_cam_folder_name)

    # this part is for test the effectiveness of the class activation map
    if testonly:
        if ckpt == None:
            raise Exception("No checkpoint model is provided")
        
        # create cam model
        if useresnet:
            net_cam = network.wideResNet_cam(num_class=num_class)
        else:
            net_cam = network.scalenet101_cam(structure_path='network/structures/scalenet101.json', num_class=num_class)
            
        model_path = "modelstates/" + ckpt + ".pth"
        pretrained = torch.load(model_path)['model']
        pretrained = {k[7:]: v for k, v in pretrained.items()}
        pretrained['fc1.weight'] = pretrained['fc1.weight'].unsqueeze(-1).unsqueeze(-1).to(torch.float64)
        net_cam.load_state_dict(pretrained)
            
        net_cam = torch.nn.DataParallel(net_cam, device_ids=devices).cuda()
        print("successfully load model states.")
        
        # calculate MIOU
        generate_validation_cam(net_cam, 224, batch_size, resize, validation_dataset_path, validation_cam_folder_name, ckpt, scales, elimate_noise=True, label_path=f'groundtruth.json', majority_vote=False)
        valid_image_path = f'{target_dataset}_valid_out_cam/{ckpt}'
        valid_iou = get_overall_valid_score(valid_image_path, num_workers=8)
        print(f"test mIOU score is: {valid_iou}")
        exit()

    if model_name == None:
        raise Exception("Model name is not provided for the traning phase!")
    # load model
    prefix = ""
    if useresnet:
        prefix = "resnet"
        resnet38_path = "weights/res38d.pth"
        if adl_drop_rate == 0:
            print("Original Network used.")
            net = network.wideResNet(regression_activate=activate_regression, num_class=num_class)
            # print(net.state_dict().keys())
        else:
            print("adl network used!")
            net = network.wideResNet(adl_drop_rate=adl_drop_rate, adl_threshold=adl_threshold, regression_activate=activate_regression, num_class=num_class)
        net.load_state_dict(torch.load(resnet38_path), strict=False)
    else:
        prefix = "scalenet"
        if adl_drop_rate == 0:
            print("Original Network used.")
            net = network.scalenet101(structure_path='network/structures/scalenet101.json', ckpt='weights/scalenet101.pth', num_class=num_class, regression_activate=activate_regression)
        else:
            print("adl network used!")
            net = network.scalenet101(structure_path='network/structures/scalenet101.json', ckpt='weights/scalenet101.pth', num_class=num_class, adl_drop_rate=adl_drop_rate, adl_threshold=adl_threshold, regression_activate=activate_regression)
    net = torch.nn.DataParallel(net, device_ids=devices).cuda()
    
    # data augmentation
    scale = (0.7, 1)
    if rand_aug:
        tfm = rand_augment_transform(
            config_str='rand-m9-n3-mstd0.5', 
            hparams={'translate_const': 60, 'img_mean': (124, 116, 104)}
        )
        train_transform = transforms.Compose([
            tfm,
            transforms.RandomResizedCrop(size=resize, scale=scale),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            # transforms.ToTensor(),
            transforms.Normalize(mean=[0.678,0.505,0.735], std=[0.144,0.208,0.174])
        ])
    else:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=resize, scale=scale),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            # transforms.ToTensor(),
            transforms.Normalize(mean=[0.678,0.505,0.735], std=[0.144,0.208,0.174])
        ])

    #cutmix init
    if cutmix_alpha == 0:
        print("cutmix not enabled!")
        cutmix_fn = None
    else:
        print("cutmix enabled!")
        cutmix_fn = Mixup(mixup_alpha=0, cutmix_alpha=cutmix_alpha,
                        cutmix_minmax=[0.4, 0.8], prob=1, switch_prob=0, 
                        mode="single", correct_lam=True, label_smoothing=0.0,
                        num_classes=3)

    # load training dataset
    TrainDataset = dataset.OriginPatchesDataset(data_path_name='Dataset_warwick/1.training/img', transform=train_transform, cutmix_fn=cutmix_fn)
    print("train Dataset", len(TrainDataset))
    TrainDatasampler = torch.utils.data.RandomSampler(TrainDataset)
    TrainDataloader = DataLoader(TrainDataset, batch_size=batch_size, num_workers=4, sampler=TrainDatasampler, drop_last=True)

    # optimizer and loss
    optimizer = PolyOptimizer(net.parameters(), base_lr, weight_decay=1e-4, max_step=epochs, momentum=0.9)
    criteria = torch.nn.BCEWithLogitsLoss(reduction='mean') # pos_weight=torch.tensor([0.73062968, 0.65306307, 2.11971588])
    regression_criteria = torch.nn.MSELoss(reduction='mean').cuda()
    criteria.cuda()

    # train loop
    loss_t = []
    iou_v = []
    best_val = 0
    
    for i in range(epochs):
        count = 0
        running_loss = 0.
        net.train()

        for img, label, area in tqdm(TrainDataloader):
            count += 1
            img = img.cuda()
            label = label.cuda()

            if activate_regression:
                area = area.cuda()
                scores, predarea = net(img)
                w = torch.sum(area, dim=1)
                w[w < 0] = 0

                regression_loss = 0.
                for index in torch.where(w!=0)[0]:
                    regression_loss += regression_criteria(predarea[index][None, :], area[index][None, :])
                regression_loss = regression_loss / len(torch.where(w!=0)[0])

                loss = criteria(scores, label.float()) + 0.5 * regression_loss
            
            else:
                scores = net(img)
                loss = criteria(scores, label.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        train_loss = running_loss / count
        loss_t.append(train_loss)

        valid_iou = 0
        if test_every != 0 and ((i + 1) % test_every == 0 or (i + 1) == epochs):
            if useresnet:
                net_cam = network.wideResNet_cam()
            else:
                net_cam = network.scalenet101_cam(structure_path='network/structures/scalenet101.json')

            pretrained = net.state_dict()
            pretrained = {k[7:]: v for k, v in pretrained.items()}
            pretrained['fc1.weight'] = pretrained['fc1.weight'].unsqueeze(-1).unsqueeze(-1).to(torch.float64)
            if 'fcregression.weight' in pretrained: 
                del pretrained['fcregression.weight']
                del pretrained['fcregression.bias']
            net_cam.load_state_dict(pretrained)
            net_cam = torch.nn.DataParallel(net_cam, device_ids=devices).cuda()

            # calculate MIOU
            valid_image_path = os.path.join(validation_cam_folder_name, model_name)
            if target_dataset == 'wsss':
                generate_validation_cam(net_cam, 224, batch_size, resize, validation_dataset_path, validation_cam_folder_name, model_name, scales, elimate_noise=True, label_path=f'groundtruth.json', majority_vote=False, is_valid=True)
                valid_iou = get_overall_valid_score(valid_image_path, 'Dataset_wsss/2.validation/mask', num_workers=8, mask_path='Dataset_wsss/2.validation/background-mask')
            elif target_dataset == 'warwick':
                generate_validation_cam(net_cam, 224, batch_size, resize, validation_dataset_path, validation_cam_folder_name, model_name, scales)
                valid_iou = get_overall_valid_score(valid_image_path, 'Dataset_warwick/2.validation/mask', num_workers=8)
            iou_v.append(valid_iou)
            
            if valid_iou > best_val:
                print("Updating the best model..........................................")
                best_val = valid_iou
                torch.save({"model": net.state_dict(), 'optimizer': optimizer.state_dict()}, "./modelstates/" + prefix + "_" + model_name + "_best.pth")
        
        print(f'Epoch [{i+1}/{epochs}], Train Loss: {train_loss:.4f}, Valid mIOU: {valid_iou:.4f}')

        # if save_every != 0 and (i + 1) % save_every == 0 and (i + 1) != epochs:
        #     torch.save({"model": net.state_dict(), 'optimizer': optimizer.state_dict()}, "./modelstates/" + prefix + "_" + model_name + "_ep" + str(i+1) + ".pth")

    torch.save({"model": net.state_dict(), 'optimizer': optimizer.state_dict()}, "./modelstates/" + prefix + "_" + model_name + "_last.pth")

    plt.figure(1)
    plt.plot(loss_t)
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.title('train loss')
    plt.savefig('./result/train_loss.png')
    plt.close()

    # plt.figure(2)
    # plt.plot(regression_loss_track, label="reg loss")
    # plt.legend()
    # plt.ylabel('loss value')
    # plt.xlabel('batchs*5')
    # plt.title('loss')
    # plt.savefig('./result/loss_trackadl.png')

    plt.figure(3)
    plt.plot(iou_v)
    plt.ylabel('accuracy')
    plt.xlabel('epochs')
    plt.title('valid accuracy')
    plt.savefig('./result/valid_iou.png')
