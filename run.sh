# You can modify this file to make all the process run here in sequence

CUDA_VISIBLE_DEVICES=1 python generate_valid_CAM.py
CUDA_VISIBLE_DEVICES=1 python generate_submit.py -v
CUDA_VISIBLE_DEVICES=1 python visualization_compare.py