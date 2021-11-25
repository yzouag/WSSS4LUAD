import argparse
from utils import crop_subpatches

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--side_length', default=96, type=int, help='crop size')
    parser.add_argument('--stride', type=int, default=32, help='crop step size')
    parser.add_argument('--white_threshold', type=float, default=0.9, help='the propotion of white pixels that will let us view this subpatch as invalid and abandon this cropped subpatch')
    parser.add_argument('--cell_percent', type=float, default=0.1, help='the threshold for asserting one image contain certain label for the validation set')
    args = parser.parse_args()

    side_length = args.side_length
    stride = args.stride
    white_threshold = args.white_threshold
    cell_percentage = args.cell_percent

    print('crop valid images to small patches...')
    crop_subpatches.crop_valid_set(side_length, stride, white_threshold, cell_percentage)
    print('crop train images to small patches...')
    crop_subpatches.crop_train_set(white_threshold, side_length, stride)