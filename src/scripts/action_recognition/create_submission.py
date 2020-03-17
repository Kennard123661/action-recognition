import os
import argparse
import torch
import numpy as np

if __name__ == '__main__':
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from scripts import set_determinstic_mode
from data.breakfast import get_submission_segments
from config import ROOT_DIR
SUBMISSION_DIR = os.path.join(ROOT_DIR, 'submissions', 'action-recognition')


def _parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-c', '--config', required=True, type=str, help='config filename e.g -c base')
    argparser.add_argument('-m', '--model', required=True, type=str, help='model e.g. baseline')
    argparser.add_argument('-d', '--device', default=0, choices=np.arange(torch.cuda.device_count()),
                           type=int, help='device to run on')
    return argparser.parse_args()


def main():
    set_determinstic_mode()
    args = _parse_args()

    if args.model == 'baselines':
        from scripts.action_recognition.train_baseline import Trainer
    else:
        raise ValueError('no such model')
    submission_dir = os.path.join(SUBMISSION_DIR, args.model)
    if not os.path.exists(submission_dir):
        os.makedirs(submission_dir)
    submission_file = os.path.join(submission_dir, args.config + '.csv')
    if os.path.exists(submission_file):
        raise ValueError(submission_file + ' exists, please delete if you want a new submission with this name')

    trainer = Trainer(args.config, args.device)
    submssion_segments = get_submission_segments()
    predictions = trainer.predict(submssion_segments)
    with open(submission_file, 'w') as f:
        f.write('Id,Category\n')
        for i, prediction in enumerate(predictions):
            prediction_str = '{0},{1}\n'.format(i, prediction)
            f.write(prediction_str)


if __name__ == '__main__':
    main()
