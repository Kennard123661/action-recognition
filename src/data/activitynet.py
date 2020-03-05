import os
from . import DATA_DIR
DATASET_DIR = os.path.join(DATA_DIR, 'activitynet')
SPLIT_DIR = os.path.join(DATASET_DIR, 'splits')

VIDEO_DIR = os.path.join(DATASET_DIR, 'videos')
ANET13_ANNOTATION_JSON = os.path.join(SPLIT_DIR, 'activity_net.v1-3.min.json')
ANET13_DOWNLOAD_SCRIPT = os.path.join(DATASET_DIR, 'download-anet13.sh')

EXTRACTED_DIR = os.path.join(DATASET_DIR, 'extracted')
