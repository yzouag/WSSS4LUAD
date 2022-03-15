# Weakly Supervised Learning for Whole Slide Image Segmentation
## Introduction

This is a Final Year Project of HKUST CSE department down by Yiduo Yu, Yiwen Zou, Tianqi Xiang and guided by PHD candidate Yi Li and professor Xiaomeng Li. Here is the project website: [Grand Challenge Website](https://wsss4luad.grand-challenge.org/WSSS4LUAD/).
## baseline
https://arxiv.org/pdf/2110.08048.pdf

## data
Now we are testing our model on three different dataset, they are:
- [https://wsss4luad.grand-challenge.org/WSSS4LUAD/](WSSS4LUAD)
- [https://warwick.ac.uk/fac/cross_fac/tia/data/glascontest/](GlaS (Gland Segmentation in Colon Histology Images Challenge))
- [https://warwick.ac.uk/fac/cross_fac/tia/data/mildnet/](Colorectal Adenocarcinoma Gland (CRAG) Dataset)

## project structure

```bash
├─network # backbone models
│  └─structures # scalenet structures
├─result # images and logs for experiment results
├─weights # pretrained model weights
├─utils # directory for helper functions
│  ├─metric.py # F1 score, mIOU, Dice
│  ├─pyutils.py # most helper functions located
│  ├─mixup.py # cutmix function
│  ├─torchutils.py # pytorch helper functions
│  ├─post_processing.py # post processing file
│  └─generate_CAM.py # cam generation file
├─prepare_cls_inputs.py # preprocess images for classification model (crop images, adjust validation gt)
├─prepare_seg_inputs.py # generate intermediate pseudo-mask labels
├─dataset.py # definition of Dataset and Dataloader
└─main.py # train for a classification model to generate CAM
```

The model pretrained weights could be downloaded from [Scalenet101 weights](https://pan.baidu.com/share/init?surl=NOjFWzkAVmMNkZh6jIcMzA) with extract code: f1c5 and [Resnet38 weights](https://onedrive.live.com/?authkey=%21ACgB0g238YxuTxs&id=B9423297729DF909%21106&cid=B9423297729DF909), all weights should be put under the `weights` folder.

**Now the repository only contains the first classification stage model. The segmentation model will be incorporated from another repo very soon.**
