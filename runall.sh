# You can modify this file to make all the process run here in sequence
# For validation first
python generate_CAM.py -v -side 84 -stride 28 -m cemodel_last
python cam_visualize.py -v

# for test set
python generate_CAM.py -side 84 -stride 28 -m cemodel_last
python cam_visualize.py 