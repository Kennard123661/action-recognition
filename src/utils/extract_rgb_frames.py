import argparse
import os
import numpy as np
import multiprocessing

import data.activitynet as anet
import data.breakfast as breakfast
import data.kinetics400 as kinetics

N_PROCESSED = 0


def _extract_video_frames(video_file, image_dir, n_frame_file):
    raise NotImplementedError


def _extract_dataset_frames(video_files, extracted_image_dirs, n_frame_files, n_workers):
    pass


def _parse_args():
    parser = argparse.ArgumentParser(description="extract optical flows")
    parser.add_argument("--data", default='breakfast', choices=['breakfast', 'activitynet', 'kinetics'])
    parser.add_argument("--n_workers", default=4, type=int)
    return parser.parse_args()


def main():
    args = _parse_args()
    if args.data == 'activitynet':
        video_dir = anet.VIDEO_DIR
        extracted_image_dir = anet.EXTRACTED_IMAGES_DIR
        n_video_frames_dir = anet.N_VIDEO_FRAMES_DIR
    elif args.data == 'breakfast':
        video_dir = breakfast.VIDEO_DIR
        extracted_image_dir = breakfast.EXTRACTED_IMAGES_DIR
        n_video_frames_dir = breakfast.N_VIDEO_FRAMES_DIR
    else:
        raise ValueError('no such data set')

    if not os.path.exists(extracted_image_dir):
        os.makedirs(extracted_image_dir)
    if not os.path.exists(n_video_frames_dir):
        os.makedirs(n_video_frames_dir)

    all_videos = os.listdir(video_dir)
    processed_videos = os.listdir(n_video_frames_dir)
    processed_videos = [file[:-4] for file in processed_videos]

    videos = np.setdiff1d(all_videos, processed_videos)
    if len(videos) == 0:
        print('INFO: all {} videos have been processed'.format(args.data))
        return

    video_files = [os.path.join(video_dir, video) for video in videos]
    extracted_image_dirs = [os.path.join(extracted_image_dir, video) for video in videos]
    n_frame_files = [os.path.join(n_video_frames_dir, video + '.npy') for video in videos]
    _extract_dataset_frames()


if __name__ == '__main__':
    main()
