import os
import random
import torch
import tensorflow as tf
import numpy as np
ROOT_DIR = os.path.join(os.path.dirname(__file__), '..', '..')
BASE_CHECKPOINT_DIR = os.path.join(ROOT_DIR, 'checkpoints')
BASE_LOG_DIR = os.path.join(ROOT_DIR, 'logs')
BASE_CONFIG_DIR = os.path.join(ROOT_DIR, 'configs')


def set_determinstic_mode(seed=1234):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

