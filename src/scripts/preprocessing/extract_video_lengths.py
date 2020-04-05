import os
import h5py
import cv2
import numpy as np
from utils.video_reader import get_video_fps
from p_tqdm import p_umap

FRAME_BATCH_SIZE = 100


def extract_video_lengths(video_file, save_file):
    """ stores video frames as h5 files for easy access """
    cap = cv2.VideoCapture(video_file)
    n_frames = 0
    im_size = None
    while cap.isOpened():
        has_frame, frame = cap.read()
        if has_frame:
            n_frames += 1
            im_size = im_size if im_size is not None else frame.shape
        else:
            break
    cap.release()
    assert n_frames > 0
    np.save(save_file, n_frames)


def _extract_video_lengths(dset_name):
    if dset_name == 'breakfast':
        import data.breakfast as dset
    else:
        raise ValueError('no such dataset')

    video_length_dir = dset.VIDEO_LENGTHS_DIR
    if not os.path.exists(video_length_dir):
        os.makedirs(video_length_dir)

    video_dir = dset.VIDEO_DIR
    videos = os.listdir(video_dir)
    processed_videos = os.listdir(video_length_dir)
    processed_videos = [video[:-4] for video in processed_videos]  # remove the '.npy' extension
    unprocessed_videos = np.setdiff1d(videos, processed_videos)

    video_files = [os.path.join(video_dir, video) for video in unprocessed_videos]
    video_length_files = [os.path.join(video_length_dir, video + '.npy') for video in unprocessed_videos]
    p_umap(extract_video_lengths, video_files, video_length_files)


if __name__ == '__main__':
    _extract_video_lengths(dset_name='breakfast')
