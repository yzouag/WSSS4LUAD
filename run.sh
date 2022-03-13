#!/bin/bash

echo choose dataset $1
# GlaS dataset
if [ $1 == glas ]
then
    python prepare_cls_inputs.py -d glas
    CUDA_VISIBLE_DEVICES=0 python main.py -d 0 -m glas_11256 -resnet -dataset glas -test_every 1 -epoch 2
    CUDA_VISIBLE_DEVICES=0 python prepare_seg_inputs.py -d 0 -dataset glas -ckpt resnet_glas_11256_best 

# WSSS4LUAD dataset
elif [ $1 == luad ]
then
    python prepare_cls_inputs.py -d luad
    CUDA_VISIBLE_DEVICES=0 python main.py -d 0 -m luad_224_75 -resnet -dataset luad -test_every 5 -epoch 35
    CUDA_VISIBLE_DEVICES=0 python prepare_seg_inputs.py -d 0 -dataset luad -ckpt resnet_luad_22475_best 

# CRAG dataset
else
    CUDA_VISIBLE_DEVICES=2 python main.py -d 0 -m glas_112112 -resnet -dataset warwick -test_every 4 -epoch 40
fi