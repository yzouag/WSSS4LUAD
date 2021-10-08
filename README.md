# WSSS4LUAD

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
- `dataset.py` The dataset file contains all the useful dataset
- `legacy/` All the deprecated code
- `image/` The accuracy and loss curves of the trained models
- `utils/util.py` All the general purpose functions