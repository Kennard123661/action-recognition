import os
import argparse
import torch
import numpy as np

if __name__ == '__main__':
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from scripts import set_determinstic_mode
import data.breakfast as breakfast
from config import ROOT_DIR
from scripts.submission_utils import _print_submission_accuracy
SUBMISSION_DIR = os.path.join(ROOT_DIR, 'submissions', 'action-segmentation')


def _parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-c', '--config', required=True, type=str, help='config filename e.g -c base')
    return argparser.parse_args()


def main():
    set_determinstic_mode()
    args = _parse_args()

    submission_dir = os.path.join(SUBMISSION_DIR, 'zhangcan', args.config)
    submission_feats, _, _ = breakfast.get_mstcn_data(split='test')

    frame_prediction_dir = submission_dir
    video_names = [os.path.split(feat_file)[-1] for feat_file in submission_feats]
    video_names = [feat_file.split('.')[0] for feat_file in video_names]
    mapping_dict = breakfast.read_mapping_file()
    frame_level_predictions = []
    for i, video_name in enumerate(video_names):
        frame_prediction_file = os.path.join(frame_prediction_dir, video_name)
        with open(frame_prediction_file, 'r') as f:
            frame_predictions = f.readlines()[1]
        frame_predictions = frame_predictions.strip().split(' ')
        frame_predictions = [mapping_dict[prediction] for prediction in frame_predictions]
        frame_level_predictions.append(frame_predictions)

    with open(breakfast.SUBMISSION_LABEL_FILE, 'r') as f:
        submission_timestamps = f.readlines()
    submission_timestamps = [line.strip().split(' ') for line in submission_timestamps]
    submission_timestamps = [np.array(timestamps).astype(int) for timestamps in submission_timestamps]

    n_segments = 0
    submission_str = 'Id,Category\n'
    for i, video_name in enumerate(video_names):
        video_timestamps = submission_timestamps[i]
        n_timestamps = len(video_timestamps)
        video_frame_predictions = frame_level_predictions[i]
        for j in range(n_timestamps-1):
            start = video_timestamps[j]
            end = video_timestamps[j+1]
            segment_frame_predictions = video_frame_predictions[start:end]
            counts = np.bincount(segment_frame_predictions)

            segment_prediction = np.argmax(counts).item()
            submission_str += '{0},{1}\n'.format(n_segments, segment_prediction)
            n_segments += 1

    submission_file = os.path.join(submission_dir, 'submission.csv')
    with open(submission_file, 'w') as f:
        f.write(submission_str)
    _print_submission_accuracy(submission_file)


if __name__ == '__main__':
    main()
