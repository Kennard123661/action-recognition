import os
import numpy as np

filepath = '/home/kennardngpoolhua/Downloads/segment.txt'

N_GT_SEGMENTS = 1284


def check_segment_txt():
    with open(filepath, 'r') as f:
        segments = f.readlines()

    # remove first line
    segments = [segment.strip().split(' ') for segment in segments]
    n_video_segments = [len(segment) - 1 for segment in segments]
    n_segments = np.sum(n_video_segments)
    print('INFO: there are {} segments'.format(n_segments))


def main():
    check_segment_txt()


if __name__ == '__main__':
    main()
