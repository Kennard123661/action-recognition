import os
import numpy as np
from data import DATA_DIR
from tqdm import tqdm
from PIL import Image

DATASET_DIR = os.path.join(DATA_DIR, 'breakfast')
VIDEO_DIR = os.path.join(DATASET_DIR, 'videos')
NEW_LABEL_DIR = os.path.join(DATASET_DIR, 'new-labels')
EXTRACTED_DIR = os.path.join(DATASET_DIR, 'extracted')

COARSE_LABELS = ['cereals' 'coffee' 'friedegg' 'juice' 'milk' 'pancake' 'salat' 'sandwich', 'scrambledegg' 'tea']
EXTRACTED_IMAGES_DIR = os.path.join(EXTRACTED_DIR, 'images')
N_VIDEO_FRAMES_DIR = os.path.join(DATASET_DIR, 'n-video-frames')
LABEL_DIR = os.path.join(DATASET_DIR, 'labels')

SPLIT_DIR = os.path.join(DATASET_DIR, 'splits')
TRAIN_SPLIT_FILE = os.path.join(SPLIT_DIR, 'train.split1.bundle')
TEST_SPLIT_FILE = os.path.join(SPLIT_DIR, 'test.split1.bundle')

OLD_I3D_DIR = os.path.join(DATASET_DIR, 'old-i3d')
I3D_DIR = os.path.join(DATASET_DIR, 'i3d')
I3D_2048_DIR = os.path.join(DATASET_DIR, 'i3d-2048')
MAPPING_FILE = os.path.join(SPLIT_DIR, 'mapping.txt')

PROVIDED_GT_DIR = os.path.join(DATASET_DIR, 'provided-gt')
MSTCN_DIR = os.path.join(DATASET_DIR, 'mstcn')
MSTCN_LABEL_DIR = os.path.join(MSTCN_DIR, 'groundTruth')
MSTCN_SEGMENT_LABEL_DIR = os.path.join(MSTCN_DIR, 'segment-labels')
MSTCN_FEATURE_DIR = os.path.join(MSTCN_DIR, 'features')

SUBMISSION_LABEL_FILE = os.path.join(DATASET_DIR, 'test_segment.txt')
SUBMISSION_GT_FILE = os.path.join(DATASET_DIR, 'test-segment-gt.txt')
MEAN_FILE = os.path.join(DATASET_DIR, 'mean.npy')
STD_FILE = os.path.join(DATASET_DIR, 'std.npy')
N_CLASSES = 48
N_MSTCN_CLASSES = 48

TENSOR_MEAN = [0.42384474, 0.39556269, 0.34748514]
TENSOR_STD = [0.15591848, 0.14713841, 0.13312177]


def read_mapping_file():
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
    mapping_dict = read_mapping_file()
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


def read_i3d_data(videoname, i3d_length, window=None):
    videoname = videoname[:-4]
    if i3d_length == 400 or i3d_length == 2048:
        if i3d_length == 400:
            i3d_dir = I3D_DIR
        elif i3d_length == 2048:
            i3d_dir = I3D_2048_DIR
        else:
            raise ValueError('no such feature length')
        i3d_file = os.path.join(i3d_dir, videoname + '.npy')
        i3d_feats = np.load(i3d_file).astype(np.float32)
        if window is not None:
            start, end = window
            i3d_feats = i3d_feats[start:end]
    elif i3d_length == (2048 + 400):
        i3d_file = os.path.join(I3D_DIR, videoname + '.npy')
        i3d_feats = np.load(i3d_file).astype(np.float32)
        if window is not None:
            start, end = window
            i3d_feats = i3d_feats[start:end]
        i3d_file = os.path.join(I3D_2048_DIR, videoname + '.npy')
        other_i3d_feats = np.load(i3d_file).astype(np.float32)
        if window is not None:
            start, end = window
            other_i3d_feats = other_i3d_feats[start:end]
        min_length = min(len(other_i3d_feats), len(i3d_feats))
        i3d_feats = i3d_feats[:min_length]
        other_i3d_feats = other_i3d_feats[:min_length]
        i3d_feats = np.concatenate([i3d_feats, other_i3d_feats], axis=1)
    else:
        raise ValueError('no such feature length')
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


def read_frame(videoname, frame_id):
    frame_dir = os.path.join(EXTRACTED_IMAGES_DIR, videoname)
    frame_file = os.path.join(frame_dir, '{0:06d}.png'.format(frame_id))
    im = Image.open(frame_file)
    return im


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


def _check_mstcn_gt():
    label_filenames = sorted(os.listdir(PROVIDED_GT_DIR))
    video_names = [filename.split('.')[0] for filename in label_filenames]
    print('INFO: checking mstcn groundtruth...')
    for i in range(len(video_names)):
        provided_gt_file = os.path.join(PROVIDED_GT_DIR, video_names[i] + '.txt')
        if not os.path.exists(provided_gt_file):
            continue
        with open(provided_gt_file, 'r') as f:
            provided_gt = f.readlines()
        provided_gt = [label.strip() for label in provided_gt]

        mstcn_label_file = os.path.join(MSTCN_LABEL_DIR, video_names[i] + '.txt')
        with open(mstcn_label_file, 'r') as f:
            predicted_labels = f.readlines()
        predicted_labels = [label.strip() for label in predicted_labels]

        for j, label in enumerate(provided_gt):
            other_label = predicted_labels[j]
            assert label == other_label


def _read_mstcn_label(video_name):
    label_file = os.path.join(MSTCN_LABEL_DIR, video_name + '.txt')
    with open(label_file, 'r') as f:
        labels = f.readlines()
    labels = [label.strip() for label in labels]
    return labels


def _generate_mstcn_segment_labels():
    """ note that 0-based indexing is used. """
    if not os.path.exists(MSTCN_SEGMENT_LABEL_DIR):
        os.makedirs(MSTCN_SEGMENT_LABEL_DIR)
    for split in ['test', 'train']:
        video_names = get_split_videonames(split)
        all_frame_labels = [_read_mstcn_label(video_name) for video_name in video_names]
        for i, video_name in enumerate(video_names):
            video_frame_labels = all_frame_labels[i]

            segment_labels = []
            start = 0
            for fi in range(len(video_frame_labels)):
                if fi == 0:
                    continue

                prev_frame_label = video_frame_labels[fi - 1]
                curr_frame_label = video_frame_labels[fi]
                if prev_frame_label != curr_frame_label:
                    segment_label = '{0} {1} {2}'.format(str(start), str(fi), prev_frame_label)
                    start = fi
                    segment_labels.append(segment_label)
            segment_labels = '\n'.join(segment_labels) + '\n'

            segment_label_file = os.path.join(MSTCN_SEGMENT_LABEL_DIR, video_name + '.txt')
            with open(segment_label_file, 'w') as f:
                f.write(segment_labels)


def get_mstcn_data(split):
    assert split in ['train', 'test']
    video_names = get_split_videonames(split)
    video_files = [os.path.join(MSTCN_FEATURE_DIR, video_name + '.npy') for video_name in video_names]
    gt_labels = [_read_mstcn_label(video_name) for video_name in video_names]

    gt_logits = []
    label_to_logit_map = read_mapping_file()
    for gt_label in gt_labels:
        gt_logit = [label_to_logit_map[label] for label in gt_label]
        gt_logits.append(gt_logit)
    assert len(gt_logits) == len(gt_labels) == len(video_files)
    return video_files, gt_labels, gt_logits


def _generate_test_segment_gt():
    video_names = get_split_videonames('test')
    with open(SUBMISSION_LABEL_FILE, 'r') as f:
        submission_timestamps = f.readlines()
    submission_timestamps = [line.strip().split(' ') for line in submission_timestamps]
    submission_timestamps = [np.array(timestamps).astype(int) for timestamps in submission_timestamps]

    gt_labels = [_read_mstcn_label(video_name) for video_name in video_names]
    segment_labels = []
    for i, video_labels in enumerate(gt_labels):
        video_timestamps = submission_timestamps[i]
        n_timestamps = len(video_timestamps)
        video_labels = gt_labels[i]
        for fi in range(n_timestamps - 1):
            start = video_timestamps[fi]
            end = video_timestamps[fi+1]
            segment_label = video_labels[start:end]
            assert len(np.unique(segment_label)) == 1, print(segment_label)
            segment_label = segment_label[0]
            segment_labels.append(segment_label)

    with open(SUBMISSION_GT_FILE, 'w') as f:
        for segment_label in segment_labels:
            f.write(str(segment_label) + '\n')


def _print_coarse_labels():
    video_names = os.listdir(VIDEO_DIR)
    video_names = [name.split('.')[0] for name in video_names]
    coarse_labels = [name.split('_')[-1] for name in video_names]
    print(np.unique(coarse_labels))


if __name__ == '__main__':
    # _check_mstcn_gt()
    # get_mstcn_data(split='train')
    # _generate_mstcn_segment_labels()
    _print_coarse_labels()
    # _generate_test_segment_gt()
    # segments = get_submission_segments()
    # print(len(segments))
    # cvt_i3d_to_numpy()
    # _read_i3d_data('P03_cam01_P03_cereals.avi')
    # print(_read_label_file(os.path.join(LABEL_DIR, 'P03.cam01.P03_cereals.avi.labels')))
    pass
