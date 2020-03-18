import os
from scripts import BASE_CONFIG_DIR, BASE_CHECKPOINT_DIR, BASE_LOG_DIR

ACTION_SEG_LOG_DIR = os.path.join(BASE_LOG_DIR, 'action-segmentation')
ACTION_SEG_CONFIG_DIR = os.path.join(BASE_CONFIG_DIR, 'action-segmentation')
ACTION_SEG_CHECKPOINT_DIR = os.path.join(BASE_CHECKPOINT_DIR, 'action-segmentation')
