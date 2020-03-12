import torch.nn as nn
import os
import json
from scripts import BASE_LOG_DIR, BASE_CHECKPOINT_DIR, BASE_CONFIG_DIR

CHECKPOINT_DIR = os.path.join(BASE_CHECKPOINT_DIR, 'baselines')
LOG_DIR = os.path.join(BASE_LOG_DIR, 'baselines')
CONFIG_DIR = os.path.join(BASE_CONFIG_DIR, 'configs')


class Trainer:
    def __init__(self, experiment):
        config_file = os.path.join(CONFIG_DIR, experiment + '.json')
        assert os.path.exists(config_file), 'config file {} does not exist'.format(config_file)
        self.experiment = experiment
        with open(config_file, 'r') as f:
            configs = json.load(f)

        self.lr = configs['lr']
        self.train_batch_size = configs['train-batch-size']
        self.test_batch_size = configs['test-batch-size']

        self.log_dir = os.path.join(LOG_DIR, experiment)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.checkpoint_dir = os.path.join(CHECKPOINT_DIR, experiment)
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        model_id = configs['model-id']
        if model_id == '1':
            pass
        else:
            raise ValueError('no such model')

        self.loss_fn = nn.CrossEntropyLoss()




