# WSSS4LUAD

## Introduction

This is a project from the [Grand Challenge Website](https://wsss4luad.grand-challenge.org/WSSS4LUAD/). Currently we achieved a mIOU score of `0.7411`, ranked 10 on the test phase [leader board](https://wsss4luad.grand-challenge.org/evaluation/test-phase/leaderboard/) of the challenge submitted with the account shichuanyexi@gmail.com. Due to intense coursework during the semester, we expect to give further improvements after the Fall 2021-2022 term (from mid-December to late-January) to a mIOU score above 0.80, this is possible since currently we still have many ideas not implemented.

## Dataset

The dataset is obtained from the [challenge website](https://wsss4luad.grand-challenge.org/Datasets/) but it is currently locked due to the close of the competition. 

Please drop an email to yyubm@connect.ust.hk or yzouag@connect.ust.hk and we will review whether we can send the dataset to you based on the request reason.

The folder structure is listed below(under the main folder WSSS4LUAD):

```
├── Dataset
│   ├── 1.training
│   ├── 2.validation
│   │   ├── background-mask
│   │   ├── img
│   │   └── mask
│   └── 3.testing
│       ├── background-mask
│       └── img
```



**Once again, this project is not finished yet and we are still in progress!**

## crop images

If you only want to get the single label images, use the command below

```bash
python crop_single.py --side_length 128 --stride 32 --white_threshold 0.9 --cell_percent 0.1
```

`--side_length` and `--stride` is the configuration of the crop patch size and stride. 

`--white_threshold` is the propotion of white pixels that will let us view this subpatch as invalid and abandon this cropped subpatch

`--cell_percent` is the threshold for asserting one image contain certain label for the validation set.

The validation result is in `./valid_patches`, the training single label patches are in `./train_single_label_patches`

For the training process of only single label patches

```bash
python train_single_patches.py -d 0 1 2 3 -m default
```

`-d` is used to add GPU numbers and `-m` is to specify the save model name, you can also tweak the batch_size, gamma, lr, etc.

make sure you downloaded the [Scalenet101 weights](https://pan.baidu.com/share/init?surl=NOjFWzkAVmMNkZh6jIcMzA) with extract code: f1c5 and put it in `weights/scalenet101.pth`

**IMPORTANT NOTES!**
When you are writing code, pay attention to the followings:

- Currently we will only use one phase training rather than two phase, the old training file are dumped to legacy for reference only and the new training file is called `train_integrated.py`
- All of the models currently using are scalenet101, it is located in the `network` folder
- Avoid hard code! Always use the argparser to modulate the hyperparameters! Be careful! There are loads of them!
- Please put other useful functions in `utils.util.py`
- Some data folder names are *strictly reserved* for specific purpose! like `train_single_patches`, do not use it for other purposes!

## Reserved folder names

- `sample_single_label_patches` it is where the doubledataset fetch the single_label images
- `sample_multiple_label_patches` it is where the doubledataset fetch the double_label images
- `train_single_label_patches` it is the originally cutted single_label images, it might be the same with sample_single_patches if we use only one-phase training
- `train_multiple_label_patches` it is the originally cutted double_label images, we need to filter out the useful labels to `sample_double_patches`
- `valid_patches` it is where small crops of validation images

## Main file explained
- `train_integrated.py` This is our main training file, which accept the doublelabeldataset and use pretrained scalenet101 to trin
- `*original_patches.py` All of these files are related to origin image label training and testing, in another word, the big patch model

```
│  .gitignore
│  dataset.py	# contains all the useful dataset
│  main.py
│  README.md
│  requirements.txt
│  run.sh
│  testing.ipynb
├─image		# accuracy and loss curves of trained models
├─legacy	# All the deprecated code
├─network
├─result
├─structures
├─train
└─utils		# All the general purpose functions
```

## Main Procedures

step 1: crop valid images to small patches

step 2: use big label network predict the crops, get the best threshold

step 3: crop train images

​    step 3.1: crop single label images

​    step 3.2: crop mixed label images

step 4: use big label network predict the mixed-label image small crops under the threshold

step 4.5 balancing the train data

step 5: train small_crop network

step 6: generate CAM

step 6.5: generate visualization result and validation

step 7: train segmentation network

step 8: make segmentation prediction

step 9: post processing

