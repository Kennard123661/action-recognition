import os
from . import DATA_DIR
DATASET_DIR = os.path.join(DATA_DIR, 'kinetics400')
SPLITS_DIR = os.path.join(DATASET_DIR, 'splits')

TRAIN_VIDEO_CSV = os.path.join(SPLITS_DIR, 'train.csv')
TEST_VIDEO_CSV = os.path.join(SPLITS_DIR, 'test.csv')
VAL_VIDEOS_CSV = os.path.join(SPLITS_DIR, 'validate.csv')

VIDEO_DIR = os.path.join(DATASET_DIR, 'videos')
TRAIN_VIDEO_DIR = os.path.join(VIDEO_DIR, 'train')
TEST_VIDEO_DIR = os.path.join(VIDEO_DIR, 'train')
VAL_VIDEO_DIR = os.path.join(VIDEO_DIR, 'validate')
