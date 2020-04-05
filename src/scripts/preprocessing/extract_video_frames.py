import os
import h5py
import cv2
import numpy as np
from utils.video_reader import get_video_fps
from p_tqdm import p_umap

FRAME_BATCH_SIZE = 100


def extract_video_frames(video_file, save_file):
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
    if n_frames == 0:
        return  # there are no frames available

    height, width, channels = im_size
    frames = []
    with h5py.File(save_file, 'w') as f:
        dout = f.create_dataset('video-frames', shape=[n_frames, height, width, channels], dtype=np.uint8)
        cap = cv2.VideoCapture(video_file)

        start = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            else:
                break

            if len(frames) >= FRAME_BATCH_SIZE:
                end = start + len(frames)
                dout[start:end, :, :, :] = np.array(frames)
                frames = []
                start = end
        cap.release()
        if len(frames) > 0:  # save the last few frames.
            end = start + len(frames)
            dout[start:end, :, :, :] = np.array(frames)
        fps = get_video_fps(video_file)
        f.attrs['video-fps'] = fps


def _extract_dset_frames(dset_name):
    if dset_name == 'breakfast':
        import data.breakfast as dset
    else:
        raise ValueError('no such dataset')

    extracted_frames_dir = dset.EXTRACTED_FRAMES_DIR
    if not os.path.exists(extracted_frames_dir):
        os.makedirs(extracted_frames_dir)

    video_dir = dset.VIDEO_DIR
    videos = os.listdir(video_dir)
    processed_videos = os.listdir(extracted_frames_dir)
    processed_videos = [video[:-5] for video in processed_videos]  # remove the '.hdf5' extension
    unprocessed_videos = np.setdiff1d(videos, processed_videos)

    video_files = [os.path.join(video_dir, video) for video in unprocessed_videos]
    extracted_files = [os.path.join(extracted_frames_dir, video + '.hdf5') for video in unprocessed_videos]
    p_umap(extract_video_frames, video_files, extracted_files)


if __name__ == '__main__':
    raise NotImplementedError('do not run this... it takes up too much memory')
    _extract_dset_frames(dset_name='breakfast')
