# Weakly Supervised Learning for Whole Slide Image Segmentation
## Introduction

This is a Final Year Project of HKUST CSE department down by Yiduo Yu, Yiwen Zou, Tianqi Xiang and guided by PHD candidate Yi Li and professor Xiaomeng Li. Here is the project website: [Grand Challenge Website](https://wsss4luad.grand-challenge.org/WSSS4LUAD/).
## Baseline
https://arxiv.org/pdf/2110.08048.pdf

## Data
Now we are testing our model on three different dataset, they are:
- [WSSS4LUAD](https://wsss4luad.grand-challenge.org/WSSS4LUAD/)
- [GlaS (Gland Segmentation in Colon Histology Images Challenge)](https://warwick.ac.uk/fac/cross_fac/tia/data/glascontest/)
- [Colorectal Adenocarcinoma Gland(CRAG) Dataset](https://warwick.ac.uk/fac/cross_fac/tia/data/mildnet/)

## Project Structure

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

## Usage

run the following bash script to start the training and testing. We recommend the GPU memory size should be larger than 10GB.

Stage I **Classification**
```bash
pip install -r requirements.txt
./run.sh [dataset name]
```

the details settings such as epochs and learning rate can be seen by:

```bash
python main.py --help
```

Stage II **Segmentation**

1. preparation
```bash
git clone OEEM
cd OEEM
ln -s [path to WSSS GlaS] glas
ln -s [path to pretrained models] models
```

2. install mmsegmentation and mmcv
```bash
pip install mmcv==1.1.5
pip install -e .
```

3. train
```bash
bash tools/dist_train.sh configs/pspnet_oeem/pspnet_wres38-d8_10k_histo.py 1 runs/oeem
```

4. inference, patch merge and evaluation
```bash
bash tools/dist_test.sh configs/pspnet_oeem/pspnet_wres38-d8_10k_histo_test.py [path to best checkpoint] 1
python tools/merge_patches.py glas/test_patches glas/test_merged 2
python tools/count_miou.py glas/test_merged [path to original val gt] 2
```

<!-- **Now the repository only contains the first classification stage model. The segmentation model will be incorporated from another repo very soon.** -->
