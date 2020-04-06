import torch.nn as nn
import os
import json
import torch
import math
import torch.utils.data as tdata
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import tensorboardX
import argparse
from utils.notify_utils import telegram_watch

if __name__ == '__main__':
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from scripts.frame_recognition import FRAME_REG_LOG_DIR, FRAME_REG_CHECKPOINT_DIR, FRAME_REG_CONFIG_DIR, \
    FrameRecDataset, get_train_videos, get_test_videos
from scripts import set_determinstic_mode
import data.breakfast as breakfast
from nets.action_reg.i3d import I3D

CHECKPOINT_DIR = os.path.join(FRAME_REG_CHECKPOINT_DIR, 'i3d')
LOG_DIR = os.path.join(FRAME_REG_LOG_DIR, 'i3d')
CONFIG_DIR = os.path.join(FRAME_REG_CONFIG_DIR, 'i3d')
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

        self.log_dir = os.path.join(LOG_DIR, experiment)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.tboard_writer = tensorboardX.SummaryWriter(log_dir=self.log_dir)

        self.checkpoint_dir = os.path.join(CHECKPOINT_DIR, experiment)
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        self.segment_length = configs['segment-length']
        self.model = I3D(num_classes=breakfast.N_CLASSES)
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
        self.n_epoch_iterations = configs['n-epoch-iterations']

    def train(self, train_videos, test_videos):
        train_dataset = FrameRecDataset(train_videos, input_size=self.input_size, frame_stride=self.frame_stride,
                                        segment_length=self.segment_length)
        test_dataset = FrameRecDataset(test_videos, input_size=self.input_size, frame_stride=self.frame_stride,
                                       segment_length=self.segment_length)
        train_val_dataset = FrameRecDataset(train_videos, input_size=self.input_size, frame_stride=self.frame_stride,
                                            segment_length=self.segment_length)

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
                                      collate_fn=train_dataset.collate_fn, pin_memory=False, num_workers=NUM_WORKERS)
        self.model.train()
        losses = []
        n_iterations = 0
        for feats, logits in tqdm(dataloader):
            if n_iterations >= self.n_epoch_iterations:
                break  # terminate.
            feats = feats.cuda(self.device)
            logits = logits.cuda(self.device)

            self.optimizer.zero_grad()
            _, feats = self.model(feats)
            loss = self.loss_fn(feats, logits)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
            n_iterations += 1
        avg_loss = np.mean(losses)
        print('INFO: at epoch {0} loss = {1}'.format(self.n_epochs, avg_loss))
        self.tboard_writer.add_scalar('loss', avg_loss, self.n_epochs)

        if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(avg_loss)

    def test_step(self, test_dataset):
        dataloader = tdata.DataLoader(test_dataset, shuffle=False, batch_size=self.test_batch_size,
                                      collate_fn=test_dataset.collate_fn, pin_memory=False, num_workers=NUM_WORKERS)
        self.model.eval()
        n_correct = 0
        n_predictions = 0
        with torch.no_grad():
            for feats, logits in tqdm(dataloader):
                feats = feats.cuda(self.device)
                logits = logits.cuda(self.device).int()

                self.optimizer.zero_grad()
                predictions, _ = self.model(feats)
                predictions = torch.argmax(predictions, dim=1).int()
                is_correct = torch.eq(predictions, logits)
                n_correct += torch.sum(is_correct)
                n_predictions += is_correct.shape[0]
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


@telegram_watch
def main():
    set_determinstic_mode()
    args = _parse_args()
    trainer = Trainer(args.config, args.device)
    train_videos = get_train_videos()
    test_videos = get_test_videos()
    trainer.train(train_videos, test_videos)


if __name__ == '__main__':
    main()
    # print(torch.hub.list("moabitcoin/ig65m-pytorch"))
    # model = torch.hub.load("moabitcoin/ig65m-pytorch", "r2plus1d_34_32_ig65m", num_classes=359, pretrained=True)
