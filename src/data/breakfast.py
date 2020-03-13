import os
import numpy as np
from data import DATA_DIR
from tqdm import tqdm

DATASET_DIR = os.path.join(DATA_DIR, 'breakfast')
VIDEO_DIR = os.path.join(DATASET_DIR, 'videos')
NEW_LABEL_DIR = os.path.join(DATASET_DIR, 'new-labels')
EXTRACTED_DIR = os.path.join(DATASET_DIR, 'extracted')

EXTRACTED_IMAGES_DIR = os.path.join(EXTRACTED_DIR, 'images')
N_VIDEO_FRAMES_DIR = os.path.join(DATASET_DIR, 'n-video-frames')
LABEL_DIR = os.path.join(DATASET_DIR, 'labels')

SPLIT_DIR = os.path.join(DATASET_DIR, 'splits')
TRAIN_SPLIT_FILE = os.path.join(SPLIT_DIR, 'train.split1.bundle')
TEST_SPLIT_FILE = os.path.join(SPLIT_DIR, 'test.split1.bundle')

OLD_I3D_DIR = os.path.join(DATASET_DIR, 'old-i3d')
I3D_DIR = os.path.join(DATASET_DIR, 'i3d')
MAPPING_FILE = os.path.join(SPLIT_DIR, 'mapping.txt')

SUBMISSION_LABEL_FILE = os.path.join(DATASET_DIR, 'test_segment.txt')
N_CLASSES = 48


def _read_mapping_file():
    with open(MAPPING_FILE, 'r') as f:
        mappings = f.readlines()
    mappings = np.array([mapping.strip().split(' ') for mapping in mappings])
    logits = mappings[:, 0].astype(int)
    actions = mappings[:, 1].astype(str)
    mapping_dict = {}
    for i, logit in enumerate(logits):
        action = actions[i]
        mapping_dict[action] = logit
    return mapping_dict


def _read_label_file(label_file):
    assert os.path.exists(label_file), 'label file {} does not exist'.format(label_file)
    with open(label_file, 'r') as f:
        labels = f.readlines()

    labels = np.array([label.strip().split(' ') for label in labels])
    windows = labels[:, 0]
    labels = labels[:, 1].astype(str)
    windows = np.array([window.split('-') for window in windows]).astype(int)
    return windows, labels


def _read_split_file(split_file):
    with open(split_file, 'r') as f:
        split_files = f.readlines()[1:]
    split_files = [line.strip() for line in split_files]
    split_files = [line.split('/')[-1] for line in split_files]
    videonames = [line[:-4] for line in split_files]
    return videonames


def get_split_videonames(split):
    if split == 'train':
        split_file = TRAIN_SPLIT_FILE
    elif split == 'test':
        split_file = TEST_SPLIT_FILE
    else:
        raise ValueError('no such split: {}'.format(split))
    return _read_split_file(split_file)


def _get_video_data(videoname):
    # all videos are have the avi extension
    videoname = videoname + '.avi'
    labelname = videoname + '.labels'

    n_video_frames_file = os.path.join(N_VIDEO_FRAMES_DIR, videoname + '.npy')
    label_file = os.path.join(LABEL_DIR, labelname)
    assert os.path.exists(label_file), 'label file {} does not exist'.format(label_file)

    # update windows
    windows, labels = _read_label_file(label_file)
    windows -= 1
    n_frames = np.load(n_video_frames_file)

    video_segments = []
    video_labels = []
    for i, (start, end) in enumerate(windows):
        end = min(end, n_frames)
        if end <= (start+1):
            break
        segment = {
            'start': start,
            'end': end,
            'video-name': videoname
        }
        video_segments.append(segment)
        video_labels.append(labels[i])
    return video_segments, video_labels


def get_data(split):
    videonames = get_split_videonames(split)
    all_segments = []
    all_labels = []
    print('INFO: retrieving {} segments'.format(split))
    for videoname in tqdm(videonames):
        segments, labels = _get_video_data(videoname)
        all_segments.extend(list(segments))
        all_labels.extend(list(labels))
    mapping_dict = _read_mapping_file()
    all_logits = [mapping_dict[label] for label in all_labels]
    return all_segments, all_labels, all_logits


def read_raw_i3d_data(videoname, window=None):
    videoname = videoname[:-4]  # remove extension
    i3d_file = os.path.join(OLD_I3D_DIR, videoname)
    with open(i3d_file, 'r') as f:
        i3d_feats = f.readlines()

    i3d_feats = [line.strip().split(' ') for line in i3d_feats]
    i3d_feats = np.array(i3d_feats).astype(np.float32)
    if window is not None:
        start, end = window
        i3d_feats = i3d_feats[start:end]
    return i3d_feats


def read_i3d_data(videoname, window=None):
    videoname = videoname[:-4]
    i3d_file = os.path.join(I3D_DIR, videoname + '.npy')
    i3d_feats = np.load(i3d_file)
    if window is not None:
        start, end = window
        i3d_feats = i3d_feats[start:end]
    return i3d_feats


def cvt_i3d_to_numpy():
    if not os.path.exists(I3D_DIR):
        os.makedirs(I3D_DIR)
    videonames = os.listdir(OLD_I3D_DIR)
    for videoname in tqdm(videonames):
        i3d_file = os.path.join(I3D_DIR, videoname + '.npy')
        if os.path.exists(i3d_file):
            continue
        i3d_feats = read_raw_i3d_data(videoname + '.avi')
        np.save(i3d_file, i3d_feats)


def get_submission_segments():
    videonames = get_split_videonames('test')
    with open(SUBMISSION_LABEL_FILE, 'r') as f:
        video_timestamps = f.readlines()
    video_timestamps = [line.strip().split(' ') for line in video_timestamps]
    video_timestamps = [np.array(timestamps).astype(int) for timestamps in video_timestamps]
    segments = []

    for i, timestamps in enumerate(video_timestamps):
        videoname = videonames[i]
        n_timestamps = len(timestamps)
        for j in range(n_timestamps - 1):
            start = timestamps[j]
            end = timestamps[j+1]
            segment = {
                'video-name': videoname + '.avi',
                'start': start,
                'end': end
            }
            segments.append(segment)
    return segments


if __name__ == '__main__':
    segments = get_submission_segments()
    print(len(segments))
    # cvt_i3d_to_numpy()
    # pass
    # _read_i3d_data('P03_cam01_P03_cereals.avi')
    # print(_read_label_file(os.path.join(LABEL_DIR, 'P03.cam01.P03_cereals.avi.labels')))
