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
import yaml
from utils.torchutils import PolyOptimizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-batch', default=20, type=int)
    parser.add_argument('-epoch', default=20, type=int)
    parser.add_argument('-lr', default=0.01, type=float)
    parser.add_argument('-test_every', default=5, type=int, help="how often to test a model while training")
    parser.add_argument('-d','--device', nargs='+', help='GPU id to use parallel', required=True, type=int)
    parser.add_argument('-m', type=str, help='the save model name')
    parser.add_argument('-resnet', action='store_true', default=False)
    parser.add_argument('-test', action='store_true', default=False)
    parser.add_argument('-ckpt', type=str, help='the checkpoint model name')
    parser.add_argument('-cutmix', type=float, default="0.0", help="alpha value of beta distribution in cutmix, 0 to disable, only test it on luad dataset to balance labels")
    parser.add_argument('-adl_threshold', type=float, default=None, help="range (0,1], the threhold for defining the salient activation values, 0 to disable")
    parser.add_argument('-adl_drop_rate', type=float, default=None, help="range (0,1], the possibility to drop the high activation areas, 0 to disable")
    parser.add_argument('-randaug', action='store_true', default=False)
    parser.add_argument('-reg', action='store_true', default=False, help="whether to use the area regression")
    parser.add_argument('-dataset', default='luad', type=str, choices=['glas', 'luad', 'crag'], help='we now support three datasets')
    args = parser.parse_args()

    batch_size = args.batch
    epochs = args.epoch
    base_lr = args.lr
    test_every = args.test_every
    devices = args.device
    model_name = args.m
    useresnet = args.resnet
    testonly = args.test
    ckpt = args.ckpt
    cutmix_alpha = args.cutmix
    adl_threshold = args.adl_threshold
    adl_drop_rate = args.adl_drop_rate
    rand_aug = args.randaug
    activate_regression = args.reg
    target_dataset = args.dataset
    if cutmix_alpha == 0:
        activate_regression = False

    with open('configuration.yml') as f:
        config = yaml.safe_load(f)
    mean = config[target_dataset]['mean']
    std = config[target_dataset]['std']
    num_class = config[target_dataset]['num_class']
    network_image_size = config['network_image_size']
    scales = config['scales']

    if not os.path.exists('modelstates'):
        os.mkdir('modelstates')
    if not os.path.exists('result'):
        os.mkdir('result')
    
    validation_folder_name = f'{target_dataset}_valid'
    validation_dataset_path = f'Dataset_{target_dataset}/2.validation/img'

    # This part is for testing the existing models in /modelstates/
    if testonly:
        if ckpt == None:
            raise Exception("No checkpoint model is provided")
        
        # Load models
        if useresnet:
            net_cam = network.wideResNet_cam(num_class=num_class)
        else:
            net_cam = network.scalenet101_cam(structure_path='network/structures/scalenet101.json', num_class=num_class)
            
        model_path = "modelstates/" + ckpt + ".pth"
        pretrained = torch.load(model_path)['model']
        pretrained = {k[7:]: v for k, v in pretrained.items()} # TODO: [7:] investigate
        pretrained['fc1.weight'] = pretrained['fc1.weight'].unsqueeze(-1).unsqueeze(-1).to(torch.float64)
        net_cam.load_state_dict(pretrained)
        net_cam = torch.nn.DataParallel(net_cam, device_ids=devices).cuda()
        print("Successfully load model states.")
        
        valid_image_path = os.path.join(validation_folder_name, model_name)
        # calculate MIOU
        if target_dataset == 'luad':
            generate_validation_cam(net_cam, config, target_dataset, batch_size, validation_dataset_path, validation_folder_name, model_name, elimate_noise=True, label_path=f'groundtruth.json', majority_vote=False)
            valid_iou = get_overall_valid_score(valid_image_path, 'Dataset_luad/2.validation/mask', num_workers=8, mask_path='Dataset_luad/2.validation/background-mask', num_class=num_class)
        # Since the validation images all contains tumor and normal tissues, so we basically could not elininate the noise for Glas and Crag dataset.
        elif target_dataset == 'glas':
            generate_validation_cam(net_cam, config, target_dataset, batch_size, validation_dataset_path, validation_folder_name, model_name)
            valid_iou = get_overall_valid_score(valid_image_path, 'Dataset_glas/2.validation/mask', num_workers=8, num_class=num_class)
        elif target_dataset == 'crag':
            generate_validation_cam(net_cam, config, target_dataset, batch_size, validation_dataset_path, validation_folder_name, model_name)
            valid_iou = get_overall_valid_score(valid_image_path, 'Dataset_crag/2.validation/mask', num_workers=8, num_class=num_class)
        
        print(f"Test mIOU score is: {valid_iou:.4f}, Valid Dice: {2 * valid_iou / (1 + valid_iou):.4f}")
        exit()

    # EXCLUSIVELY FOR TRAINING
    if model_name == None:
        raise Exception("Model name is not provided for the traning phase!")
    # load model
    if useresnet:
        prefix = "resnet"
        resnet38_path = "weights/res38d.pth"
        net = network.wideResNet(adl_drop_rate=adl_drop_rate, adl_threshold=adl_threshold, regression_activate=activate_regression, num_class=num_class)
        net.load_state_dict(torch.load(resnet38_path), strict=False)
    else:
        prefix = "scalenet"
        net = network.scalenet101(structure_path='network/structures/scalenet101.json', ckpt='weights/scalenet101.pth', num_class=num_class, adl_drop_rate=adl_drop_rate, adl_threshold=adl_threshold, regression_activate=activate_regression)

    net = torch.nn.DataParallel(net, device_ids=devices).cuda()
    
    # data augmentation
    scale = (0.7, 1)
    if rand_aug:
        tfm = rand_augment_transform(
            config_str='rand-m9-n3-mstd0.5', 
            hparams={'translate_const': 60, 'img_mean': (124, 116, 104)}
        )
    else:
        tfm = transforms.RandomHorizontalFlip(0.)

    train_transform = transforms.Compose([
            tfm,
            transforms.RandomResizedCrop(size=network_image_size, scale=scale),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            # transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    # CUTMIX CONFIGURATION
    if cutmix_alpha == 0:
        print("cutmix not enabled!")
        cutmix_fn = None
    else:
        print("cutmix enabled!")
        cutmix_fn = Mixup(mixup_alpha=0, cutmix_alpha=cutmix_alpha,
                        cutmix_minmax=[0.4, 0.8], prob=1, switch_prob=0, 
                        mode="single", correct_lam=True, label_smoothing=0.0,
                        num_classes=num_class)

    # load training dataset
    if target_dataset == 'luad':
        data_path_name = f'Dataset_{target_dataset}/1.training'
    else:
        data_path_name = f'Dataset_{target_dataset}/1.training/img'
    TrainDataset = dataset.OriginPatchesDataset(data_path_name=data_path_name, transform=train_transform, cutmix_fn=cutmix_fn, num_class=num_class)
    print("train Dataset", len(TrainDataset))
    TrainDatasampler = torch.utils.data.RandomSampler(TrainDataset)
    TrainDataloader = DataLoader(TrainDataset, batch_size=batch_size, num_workers=4, sampler=TrainDatasampler, drop_last=True)

    # optimizer and loss
    optimizer = PolyOptimizer(net.parameters(), base_lr, weight_decay=1e-4, max_step=epochs, momentum=0.9)
    criteria = torch.nn.BCEWithLogitsLoss(reduction='mean')
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
                net_cam = network.wideResNet_cam(num_class=num_class)
            else:
                net_cam = network.scalenet101_cam(structure_path='network/structures/scalenet101.json', num_class=num_class)

            pretrained = net.state_dict()
            pretrained = {k[7:]: v for k, v in pretrained.items()}
            pretrained['fc1.weight'] = pretrained['fc1.weight'].unsqueeze(-1).unsqueeze(-1).to(torch.float64)
            if 'fcregression.weight' in pretrained: 
                del pretrained['fcregression.weight']
                del pretrained['fcregression.bias']
            net_cam.load_state_dict(pretrained)
            net_cam = torch.nn.DataParallel(net_cam, device_ids=devices).cuda()

            # calculate MIOU
            valid_image_path = os.path.join(validation_folder_name, model_name)
            if target_dataset == 'luad':
                generate_validation_cam(net_cam, config, target_dataset, batch_size, validation_dataset_path, validation_folder_name, model_name, elimate_noise=True, label_path=f'groundtruth.json', majority_vote=False)
                valid_iou = get_overall_valid_score(valid_image_path, 'Dataset_luad/2.validation/mask', num_workers=8, mask_path='Dataset_luad/2.validation/background-mask', num_class=num_class)
            elif target_dataset == 'glas':
                generate_validation_cam(net_cam, config, target_dataset, batch_size, validation_dataset_path, validation_folder_name, model_name)
                valid_iou = get_overall_valid_score(valid_image_path, 'Dataset_glas/2.validation/mask', num_workers=8, num_class=num_class)
            elif target_dataset == 'crag':
                generate_validation_cam(net_cam, config, target_dataset, batch_size, validation_dataset_path, validation_folder_name, model_name)
                valid_iou = get_overall_valid_score(valid_image_path, 'Dataset_crag/2.validation/mask', num_workers=8, num_class=num_class)
            iou_v.append(valid_iou)
            
            if valid_iou > best_val:
                print("Updating the best model..........................................")
                best_val = valid_iou
                torch.save({"model": net.state_dict(), 'optimizer': optimizer.state_dict()}, "./modelstates/" + prefix + "_" + model_name + "_best.pth")
        
        print(f'Epoch [{i+1}/{epochs}], Train Loss: {train_loss:.4f}, Valid mIOU: {valid_iou:.4f}, Valid Dice: {2 * valid_iou / (1 + valid_iou):.4f}')

    torch.save({"model": net.state_dict(), 'optimizer': optimizer.state_dict()}, "./modelstates/" + prefix + "_" + model_name + "_last.pth")

    plt.figure(1)
    plt.plot(loss_t)
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.title('train loss')
    plt.savefig('./result/train_loss.png')
    plt.close()

    plt.figure(2)
    plt.plot(iou_v)
    plt.ylabel('accuracy')
    plt.xlabel('epochs')
    plt.title('valid accuracy')
    plt.savefig('./result/valid_iou.png')
