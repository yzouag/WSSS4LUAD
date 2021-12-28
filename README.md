# WSSS4LUAD

**Important Notice: We have already resumed working on the project on the `dev` branch! We have changed the pipeline greatly so please refer to the `dev` branch. Anything listed below other than the basic information could only serve as a reference.**

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

## Reserved folder names

- `valid_patches` it is where small crops of validation images
