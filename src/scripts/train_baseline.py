import torch.nn as nn
import os
import json
import torch
import math
import torch.utils.data as tdata
import torch.optim as optim
from torch.utils.data._utils.collate import default_collate
import numpy as np
import tensorboardX
from tqdm import tqdm
import argparse

from scripts import BASE_LOG_DIR, BASE_CHECKPOINT_DIR, BASE_CONFIG_DIR
from scripts import set_determinstic_mode
import data.breakfast as breakfast
from nets.action_reg import baselines

CHECKPOINT_DIR = os.path.join(BASE_CHECKPOINT_DIR, 'baselines')
LOG_DIR = os.path.join(BASE_LOG_DIR, 'baselines')
CONFIG_DIR = os.path.join(BASE_CONFIG_DIR, 'configs')

I3D_N_CHANNELS = 800
NUM_WORKERS = 8


class Trainer:
    def __init__(self, experiment, device):
        config_file = os.path.join(CONFIG_DIR, experiment + '.json')
        assert os.path.exists(config_file), 'config file {} does not exist'.format(config_file)
        self.experiment = experiment
        with open(config_file, 'r') as f:
            configs = json.load(f)
        self.device = int(device)

        self.lr = configs['lr']
        self.max_epochs = configs['max-epochs']
        self.train_batch_size = configs['train-batch-size']
        self.test_batch_size = configs['test-batch-size']
        self.n_epochs = 0
        self.n_test_segments = configs['n-test-segments']

        self.log_dir = os.path.join(LOG_DIR, experiment)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.checkpoint_dir = os.path.join(CHECKPOINT_DIR, experiment)
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        model_id = configs['model-id']
        if model_id == 'one-layer-mlp':
            self.model = baselines.OneLayerMlp(in_channels=I3D_N_CHANNELS, n_classes=breakfast.N_CLASSES)
        else:
            raise ValueError('no such model')
        self.model.cuda(self.device)
        self.loss_fn = nn.CrossEntropyLoss().cuda(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.1)
        self._load_checkpoint()

    def train(self, train_data, test_data):
        train_segments, train_labels, train_logits = train_data
        test_segments, test_labels, test_logits = test_data

        train_dataset = TrainDataset(train_segments, train_labels, train_logits)
        test_dataset = TestDataset(test_segments, test_labels, test_logits, n_samples=self.n_test_segments)

        start_epoch = self.n_epochs
        for epoch in range(start_epoch, self.max_epochs):
            self.n_epochs += 1
            self.train_step(train_dataset)
            self._save_checkpoint('model-' + str(start_epoch))
            self._save_checkpoint()  # update the latest model
            self.test_step(test_dataset)
            self.scheduler.step(epoch)

    def train_step(self, train_dataset):
        print('INFO: training at epoch {}'.format(self.n_epochs))
        dataloader = tdata.DataLoader(train_dataset, shuffle=True, batch_size=self.train_batch_size, drop_last=True,
                                      collate_fn=train_dataset.collate_fn, pin_memory=True, num_workers=NUM_WORKERS)
        self.model.train()
        losses = []
        for feats, logits in tqdm(dataloader):
            feats = feats.cuda(self.device)
            logits = logits.cuda(self.device)

            self.optimizer.zero_grad()
            feats = self.model(feats)
            loss = self.loss_fn(feats, logits)
            loss.backward()
            losses.append(loss.item())
        avg_loss = np.mean(losses)
        print('INFO: at epoch {0} loss = {1}'.format(self.n_epochs, avg_loss))

    def test_step(self, test_dataset):
        print('INFO: testing at epoch {}'.format(self.n_epochs))
        dataloader = tdata.DataLoader(test_dataset, shuffle=False, batch_size=self.test_batch_size,
                                      collate_fn=test_dataset.collate_fn, pin_memory=True, num_workers=NUM_WORKERS)
        self.model.eval()
        n_correct = 0
        n_predictions = 0
        with torch.no_grad():
            for feats, logits in tqdm(dataloader):
                feats = feats.cuda(self.device)
                logits = logits.cuda(self.device)

                feats = feats.view(-1, I3D_N_CHANNELS)
                feats = self.model(feats)
                feats = feats.view(-1, self.n_test_segments, breakfast.N_CLASSES)
                feats = torch.sum(feats, dim=1)
                predictions = torch.argmax(feats, dim=1)

                is_correct = torch.eq(predictions, logits)
                n_correct += torch.sum(is_correct)
                n_predictions += predictions.shape[0]
            accuracy = n_correct / n_predictions
        print('INFO: epoch {0} accuracy = {1}'.format(self.n_epochs, accuracy))

    def _save_checkpoint(self, checkpoint_name='model'):
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        checkpoint_file = os.path.join(self.checkpoint_dir, checkpoint_name + '.pth')
        save_dict = {
            'model': self.model.state_dict(),
            'optim': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'n-epochs': self.n_epochs
        }
        torch.save(checkpoint_file, save_dict)

    def _load_checkpoint(self, checkpoint_name='model'):
        checkpoint_file = os.path.join(self.checkpoint_dir, checkpoint_name + '.pth')
        if os.path.exists(checkpoint_file):
            print('INFO: loading checkpoint {}'.format(checkpoint_file))
            checkpoint = torch.load(checkpoint_file)
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optim'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            self.n_epochs = checkpoint['n-epochs']
        else:
            print('INFO: checkpoint does not exist, continuing...')


class TrainDataset(tdata.Dataset):
    def __init__(self, segment_dict, segment_labels, segment_logits):
        super(TrainDataset, self).__init__()
        self.segment_dicts = segment_dict
        self.segment_labels = segment_labels
        self.segment_logits = segment_logits

    def __getitem__(self, idx):
        segment_dict = self.segment_dicts[idx]
        logit = self.segment_dicts[idx]
        video_name = segment_dict['video-name']
        start, end = segment_dict['start'], segment_dict['end']

        i3d_feat = breakfast.read_i3d_data(video_name, window=[start, end])
        selected = np.random.choice(i3d_feat)
        selected = torch.from_numpy(selected)
        return selected, logit

    def __len__(self):
        return len(self.segment_dicts)

    @staticmethod
    def collate_fn(batch):
        feats, logits = zip(*batch)
        feats = torch.stack(feats)
        logits = default_collate(logits)
        return feats, logits


class TestDataset(TrainDataset):
    def __init__(self, segment_dict, segment_labels, segment_logits, n_samples=25):
        super(TrainDataset, self).__init__(segment_dict, segment_labels, segment_logits)
        self.n_samples = n_samples

    def __getitem__(self, idx):
        segment_dict = self.segment_dicts[idx]
        logit = self.segment_dicts[idx]
        video_name = segment_dict['video-name']
        start, end = segment_dict['start'], segment_dict['end']

        i3d_feat = breakfast.read_i3d_data(video_name, window=[start, end])
        sample_idxs = self._get_sample_idxs(start, end)
        selected = i3d_feat[sample_idxs]
        return selected, logit

    def _get_sample_idxs(self, start, end):
        n_frames = end - start
        if n_frames <= self.n_samples:
            sample_idxs = np.arange(start, end)
            min_dup = math.ceil(n_frames / self.n_samples)
            sample_idxs = np.repeat(sample_idxs, min_dup)
            sample_idxs = sample_idxs[:self.n_samples]
        else:
            sample_idxs = []
            n_frames = end - start
            for fid in range(n_frames):
                if (fid / n_frames) >= (len(sample_idxs) / self.n_samples):
                    sample_idxs.append(fid)
            sample_idxs = np.array(sample_idxs)
        return sample_idxs


def _parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-c', '--config', required=True, type=str, help='config filename e.g -c base')
    argparser.add_argument('-d', '--device', default=0, choices=np.arange(torch.cuda.device_count()),
                           type=int, help='device to run on')
    return argparser.parse_args()


def main():
    set_determinstic_mode()
    args = _parse_args()
    trainer = Trainer(args.config, args.device)
    train_data = breakfast.get_data('train')
    test_data = breakfast.get_data('test')
    trainer.train(train_data, test_data)
