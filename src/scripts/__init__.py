import os
import random
import torch
# import tensorflow as tf
import argparse
import numpy as np
ROOT_DIR = os.path.join(os.path.dirname(__file__), '..', '..')
BASE_CHECKPOINT_DIR = os.path.join(ROOT_DIR, 'checkpoints')
BASE_LOG_DIR = os.path.join(ROOT_DIR, 'logs')
BASE_CONFIG_DIR = os.path.join(ROOT_DIR, 'configs')
BASE_SUBMISSION_DIR = os.path.join(ROOT_DIR, 'submissions')
NUM_WORKERS = 8


def set_determinstic_mode(seed=1234):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-c', '--config', required=True, type=str, help='config filename e.g -c base')
    argparser.add_argument('-d', '--device', default=0, choices=np.arange(torch.cuda.device_count()),
                           type=int, help='device to run on')
    return argparser.parse_args()
