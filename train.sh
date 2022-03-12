CUDA_VISIBLE_DEVICES=2 python main.py -d 0 -m glas_112112 -resnet -dataset warwick -test_every 4 -epoch 40
# CUDA_VISIBLE_DEVICES=3 python prepare_seg_inputs.py -d 0 -dataset warwick -ckpt resnet_glas_11256_new_best
# python join_crops_back.py
# CUDA_VISIBLE_DEVICES=2 python main.py -d 0 -m glas_11256_new -resnet -dataset warwick -test -ckpt resnet_glas_11256_new_best