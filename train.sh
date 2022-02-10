# CUDA_VISIBLE_DEVICES=0 python main.py -d 0 -m multickpt -resnet -dataset wsss -test_every 4 -epoch 48
# CUDA_VISIBLE_DEVICES=0 python main.py -d 0 -m glas_11256 -resnet -dataset warwick -test -ckpt resnet_glas_11256_last
CUDA_VISIBLE_DEVICES=0 python prepare_seg_inputs.py -d 0 -dataset wsss -ckpt resnet_multickpt_4
CUDA_VISIBLE_DEVICES=0 python prepare_seg_inputs.py -d 0 -dataset wsss -ckpt resnet_multickpt_8
CUDA_VISIBLE_DEVICES=0 python prepare_seg_inputs.py -d 0 -dataset wsss -ckpt resnet_multickpt_12
CUDA_VISIBLE_DEVICES=0 python prepare_seg_inputs.py -d 0 -dataset wsss -ckpt resnet_multickpt_16