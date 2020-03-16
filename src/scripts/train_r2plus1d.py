import torch.nn as nn
import os
import json
import torch
import math
import torch.utils.data as tdata
import torch.optim as optim
from torch.utils.data._utils.collate import default_collate
import numpy as np
from torchvision.transforms import Compose
import torchvision.transforms._transforms_video as transforms
from tqdm import tqdm
import tensorboardX
import argparse

R2PLU1D_MODELS = ['r2plus1d_34_32_ig65m', 'r2plus1d_34_32_kinetics', 'r2plus1d_34_8_ig65m', 'r2plus1d_34_8_kinetics']

if __name__ == '__main__':
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from scripts import BASE_LOG_DIR, BASE_CHECKPOINT_DIR, BASE_CONFIG_DIR
from scripts import set_determinstic_mode
import data.breakfast as breakfast
from nets.action_reg import baselines

CHECKPOINT_DIR = os.path.join(BASE_CHECKPOINT_DIR, 'baselines')
LOG_DIR = os.path.join(BASE_LOG_DIR, 'baselines')
CONFIG_DIR = os.path.join(BASE_CONFIG_DIR, 'baselines')


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
        self.train_batch_size = configs['train-batch-size']
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

        model_id = configs['model-id']
        assert model_id in R2PLU1D_MODELS, 'model must be one of {}'.format(R2PLU1D_MODELS)
        self.n_images = int(model_id.split('_')[2])
        self.model = torch.hub.load("moabitcoin/ig65m-pytorch", model_id, num_classes=breakfast.N_CLASSES,
                                    pretrained=True)

        self.model = self.model.cuda(self.device)
        self.loss_fn = nn.CrossEntropyLoss().cuda(self.device)
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

    def train(self, train_data, test_data):
        train_segments, train_labels, train_logits = train_data
        test_segments, test_labels, test_logits = test_data

        train_dataset = TrainDataset(train_segments, train_labels, train_logits, i3d_length=self.i3d_length)
        test_dataset = TestDataset(test_segments, test_labels, test_logits, n_samples=self.n_test_segments,
                                   i3d_length=self.i3d_length)
        train_val_dataset = TestDataset(train_segments, train_labels, train_logits, i3d_length=self.i3d_length)

        start_epoch = self.n_epochs
        for epoch in range(start_epoch, self.max_epochs):
            self.n_epochs += 1
            self.train_step(train_dataset)
            self._save_checkpoint('model-{}'.format(self.n_epochs))
            self._save_checkpoint()  # update the latest model
            train_acc = self.test_step(train_val_dataset)
            test_acc = self.test_step(test_dataset)
            print('INFO: at epoch {}, the train accuracy is {} and the test accuracy is {}'.format(self.n_epochs,
                                                                                                   train_acc, test_acc))
            log_dict = {
                'train': train_acc,
                'test': test_acc
            }
            self.tboard_writer.add_scalars('accuracy', log_dict, self.n_epochs)

            if isinstance(self.scheduler, optim.lr_scheduler.StepLR):
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
            # print(feats.shape)

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

    def test_step(self, test_dataset):
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
                feats = feats.view(-1, self.n_test_segments, breakfast.N_CLASSES)
                feats = torch.sum(feats, dim=1)
                predictions = torch.argmax(feats, dim=1)

                for i, prediction in enumerate(predictions):
                    if prediction == logits[i]:
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
                feats = feats.view(-1, self.n_test_segments, breakfast.N_CLASSES)
                feats = torch.sum(feats, dim=1)
                predictions = torch.argmax(feats, dim=1)
                predictions = predictions.detach().cpu().tolist()
                all_predictions.extend(predictions)
        return all_predictions


class TrainDataset(tdata.Dataset):
    def __init__(self, segments, segment_labels, segment_logits, n_images, input_size, stride=1):
        super(TrainDataset, self).__init__()
        self.segments = segments
        self.segment_labels = segment_labels
        self.segment_logits = segment_logits
        self.n_images = int(n_images)
        self.input_size = int(input_size)
        self.stride = int(stride)

        self.transforms = Compose([
            transforms.RandomResizedCropVideo(size=input_size),
            transforms.RandomHorizontalFlipVideo(),
            transforms.ToTensorVideo(),
            transforms.NormalizeVideo(breakfast.TENSOR_MEAN, breakfast.TENSOR_STD)
        ])

    def __getitem__(self, idx):
        segment_dict = self.segments[idx]
        logit = self.segment_logits[idx]
        video_name = segment_dict['video-name']
        start, end = segment_dict['start'], segment_dict['end']
        assert start < end, '{0} has errors, logit {1}'.format(video_name, logit)

        n_frames = end - start
        if n_frames < self.n_images:
            frame_ids = np.arange(start, end).tolist()
            n_repeats = self.n_images // n_frames
            n_excess = self.n_images - n_frames * n_repeats

            frame_ids = np.concatenate([np.repeat(frame_ids[:n_excess], n_repeats+1),
                                        np.repeat(frame_ids[n_excess:], n_repeats)],
                                       axis=0)
            unique_frame_ids = np.arange(start, end)
            unique_frames = [breakfast.read_frame(video_name, frame_id) for frame_id in unique_frame_ids]
            frames = unique_frames[frame_ids - start]
        else:
            if n_frames < self.n_images * self.stride:
                max_stride = n_frames // self.n_images
            else:
                max_stride = self.stride
            max_start = n_frames - max_stride * self.n_images
            if max_start == start:
                start_frame = start
            else:
                start_frame = np.random.choice(np.arange(start, max_start))
            frame_ids = np.arange(start_frame, end, max_stride)
            frames = [breakfast.read_frame(video_name, frame_id) for frame_id in frame_ids]
        frames = torch.from_numpy(frames)
        return frames, logit

    def __len__(self):
        return len(self.segments)

    @staticmethod
    def collate_fn(batch):
        feats, logits = zip(*batch)
        feats = torch.stack(feats)
        logits = default_collate(logits)
        return feats, logits


class TestDataset(TrainDataset):
    def __init__(self, segments, segment_labels, segment_logits, i3d_length, n_samples=25):
        super(TestDataset, self).__init__(segments, segment_labels, segment_logits, i3d_length=i3d_length)
        self.n_samples = n_samples

    def __getitem__(self, idx):
        segment = self.segments[idx]
        logit = self.segment_logits[idx]
        video_name = segment['video-name']
        start, end = segment['start'], segment['end']

        i3d_feat = breakfast.read_i3d_data(video_name, window=[start, end], i3d_length=self.n_images)
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


class PredictionDataset(TestDataset):
    def __init__(self, segments, n_samples, i3d_length):
        super(PredictionDataset, self).__init__(segments=segments, segment_labels=None, segment_logits=None,
                                                n_samples=n_samples, i3d_length=i3d_length)

    def __getitem__(self, idx):
        segment = self.segments[idx]
        video_name = segment['video-name']
        start, end = segment['start'], segment['end']

        i3d_feat = breakfast.read_i3d_data(video_name, window=[start, end], i3d_length=self.n_images)
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
        # print(np.array(i3d_feats).shape)
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
    # main()
    print(torch.hub.list("moabitcoin/ig65m-pytorch"))
    model = torch.hub.load("moabitcoin/ig65m-pytorch", "r2plus1d_34_32_ig65m", num_classes=359, pretrained=True)
