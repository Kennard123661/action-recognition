import torch.nn as nn
import os
import json
import torch
import math
import torch.utils.data as tdata
import torch.optim as optim
from torch.utils.data._utils.collate import default_collate
import numpy as np
from tqdm import tqdm
import tensorboardX
import argparse
import pytorch_metric_learning.losses as metric_loss
from sklearn.metrics import pairwise_distances


if __name__ == '__main__':
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from scripts import BASE_LOG_DIR, BASE_CHECKPOINT_DIR, BASE_CONFIG_DIR
from scripts import set_determinstic_mode
import data.breakfast as breakfast
from nets.action_reg import retrieval

CHECKPOINT_DIR = os.path.join(BASE_CHECKPOINT_DIR, 'retrieval')
LOG_DIR = os.path.join(BASE_LOG_DIR, 'retrieval')
CONFIG_DIR = os.path.join(BASE_CONFIG_DIR, 'retrieval')


# I3D_N_CHANNELS = 400
NUM_WORKERS = 2


class Trainer:
    def __init__(self, experiment, device):
        config_file = os.path.join(CONFIG_DIR, experiment + '.json')
        assert os.path.exists(config_file), 'config file {} does not exist'.format(config_file)
        self.experiment = experiment
        with open(config_file, 'r') as f:
            configs = json.load(f)
        self.device = int(device)
        self.i3d_length = configs['i3d-length']

        self.lr = configs['lr']
        self.max_epochs = configs['max-epochs']
        self.test_batch_size = configs['test-batch-size']
        self.n_epochs = 0
        self.n_test_segments = configs['n-test-segments']

        self.log_dir = os.path.join(LOG_DIR, experiment)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.tboard_writer = tensorboardX.SummaryWriter(log_dir=self.log_dir)

        self.checkpoint_dir = os.path.join(CHECKPOINT_DIR, experiment)
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        self.distance_metric = configs['distance-metric']
        model_id = configs['model-id']
        self.embedding_size = configs['embedding-size']
        if model_id == 'one-layer-mlp':
            self.model = retrieval.OneLayerMlp(in_channels=self.i3d_length, embedding_size=self.embedding_size)
        elif model_id == 'three-layer-mlp':
            self.model = retrieval.ThreeLayerMlp(in_channels=self.i3d_length, embedding_size=self.embedding_size,
                                                 base_channels=configs['base-channels'])
        else:
            raise ValueError('no such model')
        self.model = self.model.cuda(self.device)
        self.loss_fn = metric_loss.MultiSimilarityLoss(alpha=2, beta=50).cuda(self.device)
        if configs['optim'] == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        elif configs['optim'] == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=configs['momentum'],
                                       nesterov=configs['nesterov'])
        else:
            raise ValueError('no such optimizer')

        if configs['scheduler'] == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=configs['lr-step'],
                                                       gamma=configs['lr-decay'])
        elif configs['scheduler'] == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min',
                                                                  patience=configs['lr-step'])
        else:
            raise ValueError('no such scheduler')
        self._load_checkpoint()

        # retrieval parameters
        self.n_samples_per_class = configs['n-class-samples']
        self.n_iterations = configs['n-iterations']
        self.distance_metric = configs['distance-metric']

    def train(self, train_data, test_data):
        train_segments, train_labels, train_logits = train_data
        test_segments, test_labels, test_logits = test_data

        train_dataset = TrainDataset(train_segments, train_labels, train_logits, i3d_length=self.i3d_length,
                                     n_iterations=self.n_iterations, n_samples_per_class=self.n_samples_per_class)
        test_dataset = TestDataset(test_segments, test_labels, test_logits, n_samples=self.n_test_segments,
                                   i3d_length=self.i3d_length)
        train_val_dataset = TestDataset(train_segments, train_labels, train_logits, i3d_length=self.i3d_length)

        start_epoch = self.n_epochs
        for epoch in range(start_epoch, self.max_epochs):
            self.n_epochs += 1
            self.train_step(train_dataset)
            self._save_checkpoint('model-{}'.format(self.n_epochs))
            self._save_checkpoint()  # update the latest model
            db_feats, db_logits = self.get_db(train_val_dataset)
            train_acc = self.test_step(db_feats, db_logits, train_val_dataset, is_train=True)
            test_acc = self.test_step(db_feats, db_logits, test_dataset, is_train=False)
            print('INFO: at epoch {}, the train accuracy is {} and the test accuracy is {}'.format(self.n_epochs,
                                                                                                   train_acc, test_acc))
            log_dict = {
                'train': train_acc,
                'test': test_acc
            }
            self.tboard_writer.add_scalars('accuracy', log_dict, self.n_epochs)

            if isinstance(self.scheduler, optim.lr_scheduler.StepLR):
                self.scheduler.step(epoch)

    def get_db(self, train_eval_dataset):
        dataloader = tdata.DataLoader(train_eval_dataset, shuffle=False, batch_size=self.test_batch_size,
                                      collate_fn=train_eval_dataset.collate_fn, pin_memory=True, num_workers=NUM_WORKERS)
        print('INFO: generating db feats')
        db_feats = []
        db_logits = []
        self.model.eval()
        with torch.no_grad():
            for feats, logits in tqdm(dataloader):
                feats = feats.cuda(self.device)
                db_logits.extend(logits.detach().cpu().tolist())
                feats = feats.view(-1, self.i3d_length)
                feats = self.model(feats)
                feats = feats.view(-1, self.n_test_segments, self.embedding_size)
                feats = torch.mean(feats, dim=1).detach().cpu().tolist()
                db_feats.extend(feats)
        return db_feats, db_logits

    def train_step(self, train_dataset):
        print('INFO: training at epoch {}'.format(self.n_epochs))
        dataloader = tdata.DataLoader(train_dataset, shuffle=True, batch_size=1, drop_last=True,
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
            self.optimizer.step()
            losses.append(loss.item())
        avg_loss = np.mean(losses)
        print('INFO: at epoch {0} loss = {1}'.format(self.n_epochs, avg_loss))
        self.tboard_writer.add_scalar('loss', avg_loss, self.n_epochs)

        if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(avg_loss)

    def test_step(self, db_feats, db_logits, test_dataset, is_train):
        dataloader = tdata.DataLoader(test_dataset, shuffle=False, batch_size=self.test_batch_size,
                                      collate_fn=test_dataset.collate_fn, pin_memory=True, num_workers=NUM_WORKERS)
        self.model.eval()
        n_correct = 0
        n_predictions = 0
        with torch.no_grad():
            for feats, logits in tqdm(dataloader):
                feats = feats.cuda(self.device)
                logits = logits.cuda(self.device)

                feats = feats.view(-1, self.i3d_length)
                feats = self.model(feats)
                feats = feats.view(-1, self.n_test_segments, self.embedding_size)
                feats = torch.mean(feats, dim=1)
                feats = feats.detach().cpu().tolist()

                predictions = []
                all_distances = pairwise_distances(feats, db_feats, metric=self.distance_metric)
                for distances in all_distances:
                    sorted_idxs = np.argsort(distances)
                    if is_train:
                        idx = sorted_idxs[1]  # take second closest, because closest is itself
                    else:
                        idx = sorted_idxs[0]
                    predictions.append(db_logits[idx])
                predictions = torch.from_numpy(np.array(predictions))

                for i, prediction in enumerate(predictions):
                    if prediction.item() == logits[i].item():
                        n_correct += 1
                n_predictions += predictions.shape[0]
            accuracy = n_correct / n_predictions
        return accuracy

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
        torch.save(save_dict, checkpoint_file)

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

    def predict(self, prediction_segments):
        dataset = PredictionDataset(prediction_segments, self.n_test_segments, i3d_length=self.i3d_length)
        dataloader = tdata.DataLoader(dataset, shuffle=False, batch_size=self.test_batch_size, num_workers=NUM_WORKERS,
                                      pin_memory=True)
        self.model.eval()
        all_predictions = []
        with torch.no_grad():
            for feats in tqdm(dataloader):
                feats = feats.cuda(self.device)
                feats = feats.view(-1, self.i3d_length)
                feats = self.model(feats)
                feats = feats.view(-1, self.n_test_segments, self.embedding_size)
                feats = torch.sum(feats, dim=1)
                predictions = torch.argmax(feats, dim=1)
                predictions = predictions.detach().cpu().tolist()
                all_predictions.extend(predictions)
        return all_predictions


class TrainDataset(tdata.Dataset):
    def __init__(self, segments, segment_labels, segment_logits, i3d_length, n_iterations, n_samples_per_class):
        super(TrainDataset, self).__init__()
        self.segments = segments
        self.segment_labels = segment_labels
        self.segment_logits = segment_logits
        self.i3d_length = int(i3d_length)
        self.n_iterations = n_iterations
        self.n_samples_per_class = n_samples_per_class

        n_labels = max(self.segment_logits) + 1
        sorted_segments = [[] for _ in range(n_labels)]
        for i, segment in enumerate(self.segments):
            logit = self.segment_logits[i]
            sorted_segments[logit].append(segment)
        self.sorted_segments = sorted_segments

    def _get_video_feats(self, segment_dict):
        start, end = segment_dict['start'], segment_dict['end']
        video_name = segment_dict['video-name']
        i3d_feat = breakfast.read_i3d_data(video_name, window=[start, end], i3d_length=self.i3d_length)
        idx = np.random.choice(np.arange(len(i3d_feat)))
        i3d_feat = torch.from_numpy(i3d_feat[idx])
        return i3d_feat

    def __getitem__(self, idx):
        sampled_feats = []
        sampled_logits = []
        for i, segments in enumerate(self.sorted_segments):
            n_segments = len(segments)
            if n_segments == 0:
                continue  # skip empty segments
            sample_idxs = np.random.choice(np.arange(n_segments), size=self.n_samples_per_class)

            for sample_idx in sample_idxs:
                sampled_segment = segments[sample_idx]
                i3d_feat = self._get_video_feats(sampled_segment)
                sampled_feats.append(i3d_feat)
            sampled_logits.extend([i] * self.n_samples_per_class)
        sampled_feats = torch.stack(sampled_feats, dim=0)
        sampled_logits = torch.from_numpy(np.array(sampled_logits))
        return sampled_feats, sampled_logits

    def __len__(self):
        return self.n_iterations

    @staticmethod
    def collate_fn(batch):
        feats, logits = zip(*batch)
        return feats[0], logits[0]


class TestDataset(tdata.Dataset):
    def __init__(self, segments, segment_labels, segment_logits, i3d_length, n_samples=25):
        super(TestDataset, self).__init__()
        self.segments = segments
        self.segment_labels = segment_labels
        self.segment_logits = segment_logits
        self.i3d_length = int(i3d_length)
        self.n_samples = n_samples

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        segment = self.segments[idx]
        logit = self.segment_logits[idx]
        video_name = segment['video-name']
        start, end = segment['start'], segment['end']

        i3d_feat = breakfast.read_i3d_data(video_name, window=[start, end], i3d_length=self.i3d_length)
        sample_idxs = self._get_sample_idxs(start, end)
        # print(i3d_feat.shape, ' ', start, ' ', end, ' ', sample_idxs)
        selected = i3d_feat[sample_idxs]
        selected = torch.from_numpy(selected)
        return selected, logit

    def _get_sample_idxs(self, start, end):
        n_frames = end - start
        if n_frames <= self.n_samples:
            sample_idxs = np.arange(n_frames)
            min_dup = math.ceil(self.n_samples / n_frames)
            sample_idxs = np.repeat(sample_idxs, min_dup, axis=0)
            sample_idxs = sample_idxs[:self.n_samples]
        else:
            sample_idxs = []
            n_frames = end - start
            for fid in range(n_frames):
                if (fid / n_frames) >= (len(sample_idxs) / self.n_samples):
                    sample_idxs.append(fid)
            sample_idxs = np.array(sample_idxs)
        sample_idxs = sample_idxs.reshape(-1)
        return sample_idxs

    @staticmethod
    def collate_fn(batch):
        feats, logits = zip(*batch)
        feats = torch.stack(feats)
        logits = default_collate(logits)
        return feats, logits


class PredictionDataset(TestDataset):
    def __init__(self, segments, n_samples, i3d_length):
        super(PredictionDataset, self).__init__(segments=segments, segment_labels=None, segment_logits=None,
                                                n_samples=n_samples, i3d_length=i3d_length)

    def __getitem__(self, idx):
        segment = self.segments[idx]
        video_name = segment['video-name']
        start, end = segment['start'], segment['end']

        i3d_feat = breakfast.read_i3d_data(video_name, window=[start, end], i3d_length=self.i3d_length)
        sample_idxs = self._get_sample_idxs(start, end)
        selected = i3d_feat[sample_idxs]
        selected = torch.from_numpy(selected)
        return selected


def _parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-c', '--config', required=True, type=str, help='config filename e.g -c base')
    argparser.add_argument('-d', '--device', default=0, choices=np.arange(torch.cuda.device_count()),
                           type=int, help='device to run on')
    return argparser.parse_args()


def _parse_split_data(split, feat_len):
    segments, labels, logits = breakfast.get_data(split)
    valid_segments = []
    valid_labels = []
    valid_logits = []
    for i, segment in enumerate(tqdm(segments)):
        start, end = segment['start'], segment['end']
        i3d_feats = breakfast.read_i3d_data(segment['video-name'], window=[start, end], i3d_length=feat_len)
        if len(i3d_feats) > 0 and 48 > logits[i] > 0:  # remove walk in and walk out.
            segment['end'] = segment['start'] + len(i3d_feats)
            valid_segments.append(segment)
            valid_labels.append(labels[i])
            valid_logits.append(logits[i])
    return [valid_segments, valid_labels, valid_logits]


def main():
    set_determinstic_mode()
    args = _parse_args()
    trainer = Trainer(args.config, args.device)
    train_data = _parse_split_data('train', trainer.i3d_length)
    test_data = _parse_split_data('test', trainer.i3d_length)
    trainer.train(train_data, test_data)


if __name__ == '__main__':
    main()
