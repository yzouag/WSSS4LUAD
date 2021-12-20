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

### backbone [`network/wide_resnet_cam.py`, `network/scalenet_cam.py`]
1.  add wide resnet-38 for fair comparision (verify the next modifications based on it, and apply scalenet after all settings are ready)

2.  multi-scale feature map fusion (concat layer2, layer3, layer4 before fc)

### training settings: [`train.py`]

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

### dataloader: [`dataset.py` (TrainSet, TestSet) ]

1. train dataset: use original pathes without further sub crop
2. train crop: add random resized crop (scale set to (0.25,1), defualt is (0.08, 1))
3. train augmentation: random augmentation for pathology
4. multi-scale test dataset: online crop for test, resize for multiple times firstly (eg, 0.5, 0.75, 1, 1.5, 2), then crop each reiszed image with stride (crop size same to train crop size)

## test and eval: [`test.py`, `eval.py`]
1. test with multi-scale test
2. use x^y for the normal channel (1 - pos.max()) to contral the foreground activation scale, and then apply argmax to get psuedo-mask, for the value of y, use grid search based on validation gt

## improvements: (`train.py`)
1. label balance: use a possitive weight in BCE loss (eg: pos_w = (neg_number / pos_number) ^ 0.5)
2. data synthesis: cut mix in one batch after augmentation (010 mix 100 -> 110, 110 mix 101 -> 111)
3. activation drop out: randomly drop high acti
