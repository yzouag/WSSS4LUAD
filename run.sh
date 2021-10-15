# You can modify this file to make all the process run here in sequence

CUDA_VISIBLE_DEVICES=1 python crop.py 
CUDA_VISIBLE_DEVICES=1 python train_integrated.py -m default -d xx -batch xx