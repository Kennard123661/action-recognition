import os
from scripts import BASE_CONFIG_DIR, BASE_CHECKPOINT_DIR, BASE_LOG_DIR, BASE_SUBMISSION_DIR
from data.breakfast import get_mstcn_data

ACTION_REG_LOG_DIR = os.path.join(BASE_LOG_DIR, 'action-recognition')
ACTION_REG_CONFIG_DIR = os.path.join(BASE_CONFIG_DIR, 'action-recognition')
ACTION_REG_CHECKPOINT_DIR = os.path.join(BASE_CHECKPOINT_DIR, 'action-recognition')
ACTION_REG_SUBMISSION_DIR = os.path.join(BASE_SUBMISSION_DIR, 'action-recognition')


def get_mstcn_action_reg_data(split):
    i3d_files, labels, logits = get_mstcn_data(split=split)
    reg_files = []
    segment_windows = []
    reg_labels = []
    reg_logits = []
    for j, label in enumerate(labels):
        feat_file = i3d_files[j]
        prev_label = label[0]
        prev_idx = 0
        label_len = len(label)
        for curr_idx in range(1, label_len):
            curr_label = label[curr_idx]
            if curr_label != prev_label:
                if prev_label != 'SIL':
                    segment_windows.append([prev_idx, curr_idx])  # start idx, end_idx
                    reg_files.append(feat_file)
                    reg_labels.append(prev_label)
                    reg_logits.append(logits[j][prev_idx])

                prev_label = curr_label
                prev_idx = curr_idx
        # add the final one
        if prev_label != 'SIL':
            segment_windows.append([prev_idx, label_len])  # start idx, end_idx
            reg_files.append(feat_file)
            reg_labels.append(prev_label)
            reg_logits.append(logits[j][prev_idx])
    assert len(reg_files) == len(segment_windows) == len(reg_labels) == len(reg_logits)
    return reg_files, segment_windows, reg_labels, reg_logits
