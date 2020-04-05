import os
import numpy as np
import h5py
from p_tqdm import p_umap

FRAME_BATCH_SIZE = 100


def _convert_label_file(raw_label_file, save_file):
    """ stores video frames as h5 files for easy access """
    with open(raw_label_file, 'r') as f:
        labels = f.readlines()
    labels = [label.strip().split(' ') for label in labels]
    labels = np.array(labels)

    timestamps = labels[:, 0]
    actions = labels[:, 1]
    timestamps = np.array([timestamp.split('-') for timestamp in timestamps]).astype(int)

    hdf5_labels = []
    for i, timestamp in enumerate(timestamps):
        action = actions[i]
        start, end = timestamp
        duration = end - start
        if end - start > 0:
            hdf5_labels.extend([action] * duration)

    hdf5_labels = [label.encode('utf8') for label in hdf5_labels]
    with h5py.File(save_file, 'w') as f:
        f.create_dataset('labels', data=hdf5_labels)


def _preprocess_video_labels(dset_name):
    if dset_name == 'breakfast':
        import data.breakfast as dset
    else:
        raise ValueError('no such dataset')

    hdf5_label_dir = dset.HDF5_LABEL_DIR
    if not os.path.exists(hdf5_label_dir):
        os.makedirs(hdf5_label_dir)

    label_dir = dset.LABEL_DIR
    labels = os.listdir(label_dir)
    processed_labels = os.listdir(hdf5_label_dir)
    processed_labels = [video[:-5] for video in processed_labels]  # remove the '.hdf5' extension
    unprocessed_labesl = np.sort(np.setdiff1d(labels, processed_labels))

    raw_label_files = [os.path.join(label_dir, video) for video in unprocessed_labesl]
    hdf5_label_files = [os.path.join(hdf5_label_dir, video + '.hdf5') for video in unprocessed_labesl]
    p_umap(_convert_label_file, raw_label_files, hdf5_label_files)


if __name__ == '__main__':
    _preprocess_video_labels(dset_name='breakfast')
