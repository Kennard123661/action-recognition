import os
from data import DATA_DIR

DATASET_DIR = os.path.join(DATA_DIR, 'breakfast')
VIDEO_DIR = os.path.join(DATASET_DIR, 'videos')
EXTRACTED_DIR = os.path.join(DATASET_DIR, 'extracted')

EXTRACTED_IMAGES_DIR = os.path.join(EXTRACTED_DIR, 'images')
N_VIDEO_FRAMES_DIR = os.path.join(DATASET_DIR, 'n-video-frames')
