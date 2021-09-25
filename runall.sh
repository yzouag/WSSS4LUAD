# You can modify this file to make all the process run here in sequence
# For validation first
# CUDA_VISIBLE_DEVICES=2 python generate_CAM.py -v -side 84 -stride 28 -m ce8428model_last
# CUDA_VISIBLE_DEVICES=2 python generate_submit.py -v

# # for test set
# CUDA_VISIBLE_DEVICES=2 python generate_CAM.py -side 84 -stride 28 -m ce8428model_last
# CUDA_VISIBLE_DEVICES=2 python generate_submit.py

CUDA_VISIBLE_DEVICES=2 python generate_valid_CAM.py
CUDA_VISIBLE_DEVICES=2 python visualization_compare.py