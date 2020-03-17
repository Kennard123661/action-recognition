import os
from scripts import BASE_CONFIG_DIR, BASE_CHECKPOINT_DIR, BASE_LOG_DIR

ACTION_REG_LOG_DIR = os.path.join(BASE_LOG_DIR, 'action-recognition')
ACTION_REG_CONFIG_DIR = os.path.join(BASE_CONFIG_DIR, 'action-recognition')
ACTION_REG_CHECKPOINT_DIR = os.path.join(BASE_CHECKPOINT_DIR, 'action-recognition')
