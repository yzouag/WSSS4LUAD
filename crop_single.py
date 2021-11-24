import argparse
from utils import crop_subpatches

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--side_length', default=96, type=int, help='crop size')
    parser.add_argument('--stride', type=int, default=32, help='crop step size')
    parser.add_argument('--white_threshold', type=float, default=0.9, help='the propotion of white pixels that will treat the crop as blank')
    args = parser.parse_args()

    side_length = args.side_length
    stride = args.stride
    white_threshold = args.white_threshold

    crop_subpatches.crop_train_set(white_threshold, side_length, stride)