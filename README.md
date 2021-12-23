# WSSS4LUAD


## baseline

https://arxiv.org/pdf/2110.08048.pdf

## data

for the training set, the average size of images is (224, 224).

## project structure

```bash
├─network # backbone models
│  └─structures # scalenet structures
├─result # images and logs for experiment results
├─train
└─utils	
```

## model design

### 1.backbone [`network/wide_resnet_cam.py`, `network/scalenet_cam.py`]
1.  add wide resnet-38 for fair comparision (verify the next modifications based on it, and apply scalenet after all settings are ready)

2.  multi-scale feature map fusion (concat layer2, layer3, layer4 before fc)

### 2.training settings: [`train.py`]

**require modification!**

#### classification

- resolution of the patches is 224 × 224
- batch size is set to 20
- training epochs is set to 20
- data augmentation: random horizontal and vertical flip with the probability 0.5. 
- learning rate of 1e − 2 with a polynomial decay policy

#### segmentation phase

- training epochs 20 
- learning rate 7e − 2
- no restriction of the image resolution
- data augmentation: horizontal and vertical flip, Gaussian blur and normalization

### 3.dataloader: [`dataset.py` (TrainSet, TestSet) ]

1. train dataset: use original pathes without further sub crop
2. train crop: add random resized crop (scale set to (0.25,1), defualt is (0.08, 1))
3. train augmentation: random augmentation for pathology
4. multi-scale test dataset: online crop for test, resize for multiple times firstly (eg, 0.5, 0.75, 1, 1.5, 2), then crop each reiszed image with stride (crop size same to train crop size)

### 4.test and eval: [`test.py`, `eval.py`]
1. test with multi-scale test
2. use x^y for the normal channel (1 - pos.max()) to control the foreground activation scale, and then apply argmax to get psuedo-mask, for the value of y, use grid search based on validation gt

### 5.improvements: (`train.py`)
1. norm: mean=[0.678,0.505,0.735] std=[0.144,0.208,0.174]
2. large model: resnest269
3. label balance: use a possitive weight in BCE loss (eg: pos_w = (neg_number / pos_number) ^ 0.5)
4. data synthesis: 4.1 original cutmix in batch; 4.2 cutmix based on label distribution (mainly stroma and tumor); 4.3 cutmix to balance different labels; 4.4 mosaic mix (eg.32x32x64 in seg, 56x56x16 cls, make sure 7 mixed types are balanced)
5. generate pseudo mask without model for single label patches (require corrosion and smoothing )
6. activation drop out: randomly drop high activation
7. area regression loss for single label patches / mixed patches in clssification (top2 loss_area)
8. diff loss weights for single-label / multi-label / mixed label
9. contrastive loss (reduce cosine distance for same category and enlarge it for the different, top2 loss_conl)
10. post-process: 10.1 drop catogory whose area < 5% in subpatch pseudo-mask generation (top3); 10.2 after bg mask, use knn to indentify small area pixels (top1); 10.3 rm small tumer and stroma in normal (top1)
