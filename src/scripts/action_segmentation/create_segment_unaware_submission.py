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
from scripts.submission_utils import get_submission_accuracy
SUBMISSION_DIR = os.path.join(ROOT_DIR, 'unaware-submissions', 'action-segmentation')


def _parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-c', '--config', required=True, type=str, help='config filename e.g -c base')
    argparser.add_argument('-m', '--model', required=True, type=str, help='model e.g. baseline')
    argparser.add_argument('-d', '--device', default=0, choices=np.arange(torch.cuda.device_count()),
                           type=int, help='device to run on')
    return argparser.parse_args()


def get_cls_results(frame_level_predictions, submission_dir, postprocess='midpoint'):
    submission_feats, _, _ = breakfast.get_mstcn_data(split='test')
    frame_prediction_dir = os.path.join(submission_dir, 'frame-level-predictions')
    if not os.path.exists(frame_prediction_dir):
        os.makedirs(frame_prediction_dir)

    video_names = [os.path.split(feat_file)[-1] for feat_file in submission_feats]
    video_names = [feat_file.split('.')[0] for feat_file in video_names]
    for i, video_name in enumerate(video_names):
        frame_prediction_file = os.path.join(frame_prediction_dir, video_name + '.txt')
        frame_predictions = np.array(frame_level_predictions[i])
        with open(frame_prediction_file, 'w') as f:
            frame_predictions = ' '.join(frame_predictions.astype(str))
            f.write(frame_predictions + '\n')

    with open(breakfast.SUBMISSION_LABEL_FILE, 'r') as f:
        submission_timestamps = f.readlines()
    submission_timestamps = [line.strip().split(' ') for line in submission_timestamps]
    submission_timestamps = [np.array(timestamps).astype(int) for timestamps in submission_timestamps]

    n_segments = 0
    submission_str = 'Id,Category\n'
    for i, video_name in enumerate(video_names):
        video_timestamps = submission_timestamps[i]
        n_gt_segments = len(video_timestamps) - 1
        video_frame_predictions = frame_level_predictions[i]

        segments = []
        prediction_len = 1
        prev_prediction = video_frame_predictions[i]
        for j in range(1, video_frame_predictions):
            curr_prediction = video_frame_predictions[j]
            if curr_prediction != prev_prediction:
                if 0 < prev_prediction < 48:
                    segments.append([prev_prediction, prediction_len])
                prev_prediction = curr_prediction
                prediction_len = 1
            else:
                prediction_len += 1

        is_selected = np.zeros(shape=len(segments))
        while len(segments) < n_gt_segments:
            segments.append(segments[-1])
            is_selected[:] = 1

        segment_lengths = np.array(segments)[:, 1]
        sorted_segment_lengths = np.flip(np.sort(np.unique(np.array(segments)[:, 1])))
        idx = 0
        while np.sum(is_selected).item() < n_gt_segments:
            max_length = sorted_segment_lengths[idx]
            is_selected[:] = segment_lengths > max_length

        selected_segments = segments[is_selected, :]
        if len(selected_segments) > n_gt_segments:
            selected_segments = selected_segments[:n_gt_segments]
        for segment in selected_segments:
            segment_prediction = segment[0]
            submission_str += '{0},{1}\n'.format(n_segments, segment_prediction)
            n_segments += 1
    submission_file = os.path.join(submission_dir, 'submission.csv')
    with open(submission_file, 'w') as f:
        f.write(submission_str)
    return get_submission_accuracy(submission_file)


def get_boundary_aware_submission_data():
    with open(breakfast.SUBMISSION_LABEL_FILE, 'r') as f:
        submission_timestamps = f.readlines()
    submission_timestamps = [line.strip().split(' ') for line in submission_timestamps]
    submission_timestamps = [np.array(timestamps).astype(int) for timestamps in submission_timestamps]

    i3d_files, _, _ = breakfast.get_mstcn_data(split='test')
    assert len(submission_timestamps) == len(i3d_files)
    return i3d_files, submission_timestamps


def main():
    set_determinstic_mode()
    args = _parse_args()

    if args.model == 'mstcn':
        from scripts.action_segmentation.train_mstcn import Trainer
    elif args.model == 'coarse-inputs':
        from scripts.action_segmentation.train_coarse_inputs import Trainer
    elif args.model == 'coarse-inputs-boundary-true':
        from scripts.action_segmentation.train_coarse_inputs_boundary_true import Trainer
    else:
        raise ValueError('no such model')
    submission_dir = os.path.join(SUBMISSION_DIR, args.model, args.config)
    if not os.path.exists(submission_dir):
        os.makedirs(submission_dir)
    else:
        raise ValueError(submission_dir + ' exists, please delete if you want a new submission with this name')

    trainer = Trainer(args.config, args.device)
    if 'boundary' in args.model:  # this is for boundary aware models
        i3d_feats, timestamps = get_boundary_aware_submission_data()
        frame_level_predictions = trainer.predict(i3d_feats, timestamps)
    else:
        submission_segments, _, _ = breakfast.get_mstcn_data(split='test')
        frame_level_predictions = trainer.predict(submission_segments)
    return get_cls_results(frame_level_predictions, submission_dir)


if __name__ == '__main__':
    main()
