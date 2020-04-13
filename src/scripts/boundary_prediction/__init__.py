import os
from scripts import BASE_CONFIG_DIR, BASE_CHECKPOINT_DIR, BASE_LOG_DIR

BOUNDARY_PRED_LOG_DIR = os.path.join(BASE_LOG_DIR, 'boundary-prediction')
BOUNDARY_PRED_CONFIG_DIR = os.path.join(BASE_CONFIG_DIR, 'boundary-prediction')
BOUNDARY_PRED_CHECKPOINT_DIR = os.path.join(BASE_CHECKPOINT_DIR, 'boundary-prediction')
