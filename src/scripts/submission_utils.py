import numpy as np
from data.breakfast import SUBMISSION_GT_FILE, read_mapping_file


def _print_submission_accuracy(submission_file):
    with open(submission_file, 'r') as f:
        predictions = f.readlines()[1:]
    predictions = [prediction.strip().split(',') for prediction in predictions]
    predictions = np.array(predictions).astype(int)[:, 1]

    label_to_logit_dict = read_mapping_file()
    with open(SUBMISSION_GT_FILE, 'r') as f:
        gt = f.readlines()
    gt = [label_to_logit_dict[line.strip()] for line in gt]
    gt = np.array(gt).astype(int)

    is_equal = np.equal(predictions, gt)
    accuracy = np.mean(is_equal)
    print('INFO: submission accuracy: {}'.format(accuracy))


if __name__ == '__main__':
    _print_submission_accuracy('/mnt/HGST6/cs5242-project/submissions/action-segmentation/mstcn/base/fake.csv')
