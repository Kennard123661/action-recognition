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
SUBMISSION_DIR = os.path.join(ROOT_DIR, 'submissions', 'action-segmentation')


def _parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-c', '--config', required=True, type=str, help='config filename e.g -c base')
    argparser.add_argument('-m', '--model', required=True, type=str, help='model e.g. baseline')
    argparser.add_argument('-d', '--device', default=0, choices=np.arange(torch.cuda.device_count()),
                           type=int, help='device to run on')
    return argparser.parse_args()


def get_cls_results(segment_predictions, submission_dir, postprocess='midpoint'):
    submission_feats, _, _ = breakfast.get_mstcn_data(split='test')
    video_names = [os.path.split(feat_file)[-1] for feat_file in submission_feats]

    with open(breakfast.SUBMISSION_LABEL_FILE, 'r') as f:
        submission_timestamps = f.readlines()
    submission_timestamps = [line.strip().split(' ') for line in submission_timestamps]
    submission_timestamps = [np.array(timestamps).astype(int) for timestamps in submission_timestamps]

    n_segments = 0
    submission_str = 'Id,Category\n'
    segment_idx = 0
    for i, video_name in enumerate(video_names):
        video_timestamps = submission_timestamps[i]
        n_timestamps = len(video_timestamps)
        for j in range(n_timestamps - 1):
            start = video_timestamps[j]
            end = video_timestamps[j + 1]

            vid_len = end - start
            assert vid_len > 0
            segment_prediction = segment_predictions[segment_idx]
            segment_idx += 1
            submission_str += '{0},{1}\n'.format(n_segments, segment_prediction)
            n_segments += 1
    assert n_segments == len(segment_predictions)

    submission_file = os.path.join(submission_dir, 'submission.csv')
    with open(submission_file, 'w') as f:
        f.write(submission_str)
    return get_submission_accuracy(submission_file)


def main():
    set_determinstic_mode()
    args = _parse_args()

    if args.model == 'mstcn':
        from scripts.action_segmentation.train_mstcn import Trainer
    elif args.model == 'coarse-inputs':
        from scripts.action_segmentation.train_coarse_inputs import Trainer
    else:
        raise ValueError('no such model')
    submission_dir = os.path.join(SUBMISSION_DIR, args.model, args.config)
    if not os.path.exists(submission_dir):
        os.makedirs(submission_dir)
    else:
        raise ValueError(submission_dir + ' exists, please delete if you want a new submission with this name')

    trainer = Trainer(args.config, args.device)
    submission_segments, _, _ = breakfast.get_mstcn_data(split='test')
    frame_level_predictions = trainer.predict(submission_segments)
    return get_cls_results(frame_level_predictions, submission_dir)


if __name__ == '__main__':
    main()
