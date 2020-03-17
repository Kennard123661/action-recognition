import torch.nn as nn
import os
import json
import torch
import torch.utils.data as tdata
import torch.optim as optim
from torch.utils.data._utils.collate import default_collate
import numpy as np
from torchvision.transforms import Compose
import torchvision.transforms._transforms_video as transforms
import torchvision.models.video as models
from tqdm import tqdm
import tensorboardX
import argparse


if __name__ == '__main__':
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from scripts.action_recognition import ACTION_REG_LOG_DIR, ACTION_REG_CHECKPOINT_DIR, ACTION_REG_CONFIG_DIR
from scripts import set_determinstic_mode
import data.breakfast as breakfast
from utils.video_utils import ToTensorVideo, ToZeroOneVideo, ResizeVideo

CHECKPOINT_DIR = os.path.join(ACTION_REG_CHECKPOINT_DIR, 'vid-models')
LOG_DIR = os.path.join(ACTION_REG_LOG_DIR, 'vid-models')
CONFIG_DIR = os.path.join(ACTION_REG_CONFIG_DIR, 'vid-models')

NORM_MEAN = [0.43216, 0.394666, 0.37645]
NORM_STD = [0.22803, 0.22145, 0.216989]
CLIP_LENGTH = 16
INPUT_SIZE = 112

NUM_WORKERS = 2


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
        self.tboard_writer = tensorboardX.SummaryWriter(log_dir=self.log_dir)

        self.checkpoint_dir = os.path.join(CHECKPOINT_DIR, experiment)
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        model_id = configs['model-id']
        if model_id == 'r3d':
            self.model = models.r3d_18(pretrained=True)
        elif model_id == 'mc3':
            self.model = models.mc3_18(pretrained=True)
        elif model_id == 'r2plus1d':
            self.model = models.r2plus1d_18(pretrained=True)
        else:
            raise ValueError('no such model')
        # replace the last layer.
        self.model.fc = nn.Linear(self.model.fc.in_features, out_features=breakfast.N_CLASSES,
                                  bias=self.model.fc.bias is not None)
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
        self.input_size = configs['input-size']
        self.frame_stride = configs['frame-stride']

    def train(self, train_data, test_data):
        train_segments, train_labels, train_logits = train_data
        test_segments, test_labels, test_logits = test_data

        train_dataset = TrainDataset(train_segments, train_labels, train_logits, frame_stride=self.frame_stride)
        test_dataset = TestDataset(test_segments, test_labels, test_logits, frame_stride=self.frame_stride,
                                   n_test_segments=self.n_test_segments)
        train_val_dataset = TestDataset(train_segments, train_labels, train_logits, frame_stride=self.frame_stride,
                                        n_test_segments=self.n_test_segments)

        start_epoch = self.n_epochs
        for epoch in range(start_epoch, self.max_epochs):
            self.n_epochs += 1
            train_acc = self.test_step(train_val_dataset)
            self.train_step(train_dataset)
            self._save_checkpoint('model-{}'.format(self.n_epochs))
            self._save_checkpoint()  # update the latest model
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
            for feats, n_segments, idxs in tqdm(dataloader):
                feats = feats.cuda(self.device)  # BN x C x T x H x W
                idxs = idxs.detach().cpu().tolist()
                logits = np.array(test_dataset.segment_logits)[idxs]

                feats = self.model(feats)  # BN x n_classes
                predictions = []
                start = 0
                for n_segment in n_segments:
                    end = start + n_segment
                    prediction = torch.sum(feats[start:end], dim=0)  # sum over the segments
                    predictions.append(prediction)
                    start = end
                predictions = torch.stack(predictions, dim=0)
                predictions = torch.argmax(predictions, dim=1)
                logits = torch.from_numpy(logits)

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
    def __init__(self, segments, segment_labels, segment_logits, frame_stride=1):
        super(TrainDataset, self).__init__()
        self.segments = segments
        self.segment_labels = segment_labels
        self.segment_logits = segment_logits
        self.frame_stride = int(frame_stride)

        self.transforms = Compose([
            ToTensorVideo(),
            ResizeVideo(INPUT_SIZE),
            transforms.RandomResizedCropVideo(size=INPUT_SIZE),
            transforms.RandomHorizontalFlipVideo(),
            ToZeroOneVideo(),
            transforms.NormalizeVideo(NORM_MEAN, NORM_STD)
        ])

    def __getitem__(self, idx):
        segment_dict = self.segments[idx]
        logit = self.segment_logits[idx]
        video_name = segment_dict['video-name']
        start, end = segment_dict['start'], segment_dict['end']
        assert start < end, '{0} has errors, logit {1}'.format(video_name, logit)

        n_frames = end - start
        if n_frames < CLIP_LENGTH:
            frame_ids = np.arange(start, end).tolist()
            n_repeats = CLIP_LENGTH// n_frames
            n_excess = CLIP_LENGTH- n_frames * n_repeats

            frame_ids = np.concatenate([np.repeat(frame_ids[:n_excess], n_repeats+1),
                                        np.repeat(frame_ids[n_excess:], n_repeats)], axis=0)
            unique_frame_ids = np.arange(start, end)
            unique_frames = [breakfast.read_frame(video_name, frame_id) for frame_id in unique_frame_ids]
            frames = []
            for frame_id in frame_ids:
                frame_idx = int(frame_id - start)
                frames.append(unique_frames[frame_idx])
        else:
            if n_frames < CLIP_LENGTH * self.frame_stride:
                max_stride = n_frames // CLIP_LENGTH
            else:
                max_stride = self.frame_stride
            max_start = n_frames - max_stride * CLIP_LENGTH
            if max_start <= start:
                start_frame = start
            else:
                start_frame = np.random.choice(np.arange(start, max_start))
            frame_ids = np.arange(start_frame, end, max_stride)[:CLIP_LENGTH]
            frames = [breakfast.read_frame(video_name, frame_id) for frame_id in frame_ids]
        frames = self.transforms(frames)
        return frames, logit

    def __len__(self):
        return len(self.segments)

    @staticmethod
    def collate_fn(batch):
        feats, logits = zip(*batch)
        feats = default_collate(feats)
        logits = default_collate(logits)
        return feats, logits


class TestDataset(TrainDataset):
    def __init__(self, segments, segment_labels, segment_logits, frame_stride=1, n_test_segments=25):
        super(TestDataset, self).__init__(segments, segment_labels, segment_logits, frame_stride)
        self.n_test_segments = n_test_segments
        self.transforms = Compose([
            ToTensorVideo(),
            ResizeVideo(INPUT_SIZE),
            transforms.CenterCropVideo(crop_size=INPUT_SIZE),
            ToZeroOneVideo(),
            transforms.NormalizeVideo(NORM_MEAN, NORM_STD)
        ])

    def _load_segments(self, video_name, start, end):
        n_frames = end - start
        if n_frames < CLIP_LENGTH:
            frame_ids = np.arange(start, end)
            n_repeats = CLIP_LENGTH // n_frames
            n_excess = CLIP_LENGTH - n_frames * n_repeats

            frame_ids = np.concatenate([np.repeat(frame_ids[:n_excess], n_repeats + 1),
                                        np.repeat(frame_ids[n_excess:], n_repeats)],
                                       axis=0)
            unique_frame_ids = np.arange(start, end)
            unique_frames = [breakfast.read_frame(video_name, frame_id) for frame_id in unique_frame_ids]
            video_segments = []
            for frame_id in frame_ids:
                frame_idx = int(frame_id - start)
                video_segments.append(unique_frames[frame_idx])
            video_segments = self.transforms(video_segments).unsqueeze(0)  # 1 x B x C x T x H x W
        else:
            if n_frames < CLIP_LENGTH * self.frame_stride:
                max_stride = n_frames // CLIP_LENGTH
            else:
                max_stride = self.frame_stride
            max_start = n_frames - max_stride * CLIP_LENGTH

            if max_start <= start:
                start_frames = [start]
            else:
                start_frames = np.arange(start, max_start)
                if len(start_frames) > self.n_test_segments:
                    sampled_start_frames = []
                    n_start_frames = len(start_frames)
                    for fi, start_frame in enumerate(start_frames):
                        if (fi / n_start_frames) >= (len(sampled_start_frames) / self.n_test_segments):
                            sampled_start_frames.append(start_frame)
                            if len(sampled_start_frames) == self.n_test_segments:
                                break
                    start_frames = np.array(sampled_start_frames)

            video_segments = []
            for start_frame in start_frames:
                frame_ids = np.arange(start_frame, end, max_stride)[:CLIP_LENGTH]
                frames = [breakfast.read_frame(video_name, frame_id) for frame_id in frame_ids]
                frames = self.transforms(frames)
                video_segments.append(frames)
            video_segments = torch.stack(video_segments, dim=0)  # N x C x T x H x W
        return video_segments

    def __getitem__(self, idx):
        segment_dict = self.segments[idx]
        logit = self.segment_logits[idx]
        video_name = segment_dict['video-name']
        start, end = segment_dict['start'], segment_dict['end']
        assert start < end, '{0} has errors, logit {1}'.format(video_name, logit)
        video_segments = self._load_segments(video_name, start, end)
        n_video_segments = video_segments.shape[0]
        return video_segments, n_video_segments, idx

    @staticmethod
    def collate_fn(batch):
        video_segments, n_video_segments, video_idxs = zip(*batch)
        video_segments = torch.cat(video_segments, dim=0)  # BN x C x T x H x W
        n_video_segments = default_collate(n_video_segments)  # B
        video_idxs = default_collate(video_idxs)  # B
        return video_segments, n_video_segments, video_idxs


class PredictionDataset(TestDataset):
    pass


def _parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-c', '--config', required=True, type=str, help='config filename e.g -c base')
    argparser.add_argument('-d', '--device', default=0, choices=np.arange(torch.cuda.device_count()),
                           type=int, help='device to run on')
    return argparser.parse_args()


def _parse_split_data(split):
    segments, labels, logits = breakfast.get_data(split)
    valid_segments = []
    valid_labels = []
    valid_logits = []
    for i, segment in enumerate(tqdm(segments)):
        start, end = segment['start'], segment['end']
        video_name = segment['video-name']
        video_dir = os.path.join(breakfast.EXTRACTED_IMAGES_DIR, video_name)
        n_video_frames = len(os.listdir(video_dir))
        end = min(n_video_frames, end)
        n_frames = end - start
        if n_frames > 0 and 48 > logits[i] > 0:  # remove walk in and walk out.
            segment['end'] = end
            valid_segments.append(segment)
            valid_labels.append(labels[i])
            valid_logits.append(logits[i])
    return [valid_segments, valid_labels, valid_logits]


def main():
    set_determinstic_mode()
    args = _parse_args()
    trainer = Trainer(args.config, args.device)
    train_data = _parse_split_data('train')
    test_data = _parse_split_data('test')
    trainer.train(train_data, test_data)


if __name__ == '__main__':
    main()
    # print(torch.hub.list("moabitcoin/ig65m-pytorch"))
    # model = torch.hub.load("moabitcoin/ig65m-pytorch", "r2plus1d_34_32_ig65m", num_classes=359, pretrained=True)
