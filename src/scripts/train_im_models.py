import torch.nn as nn
import os
import json
import torch
import math
import torch.utils.data as tdata
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data._utils.collate import default_collate
import numpy as np
from tqdm import tqdm
import tensorboardX

if __name__ == '__main__':
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from scripts import BASE_LOG_DIR, BASE_CHECKPOINT_DIR, BASE_CONFIG_DIR, NUM_WORKERS
from scripts import set_determinstic_mode, parse_args
import data.breakfast as breakfast

CHECKPOINT_DIR = os.path.join(BASE_CHECKPOINT_DIR, 'im_models')
LOG_DIR = os.path.join(BASE_LOG_DIR, 'im_models')
CONFIG_DIR = os.path.join(BASE_CONFIG_DIR, 'im_models')


class Trainer:
    def __init__(self, experiment):
        config_file = os.path.join(CONFIG_DIR, experiment + '.json')
        assert os.path.exists(config_file), 'config file {} does not exist'.format(config_file)
        self.experiment = experiment
        with open(config_file, 'r') as f:
            self.configs = json.load(f)

        self.lr = self.configs['lr']
        self.max_epochs = self.configs['max-epochs']
        self.train_batch_size = self.configs['train-batch-size']
        self.test_batch_size = self.configs['test-batch-size']
        self.n_test_segments = self.configs['n-test-segments']
        self.n_epochs = 0
        self.devices = np.arange(torch.cuda.device_count())

        self.log_dir = os.path.join(LOG_DIR, experiment)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.tboard_writer = tensorboardX.SummaryWriter(log_dir=self.log_dir)

        self.checkpoint_dir = os.path.join(CHECKPOINT_DIR, experiment)
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        model_id = self.configs['model-id']
        if model_id == 'resnet50':
            model = models.resnet50(pretrained=False)
            n_features = model.fc.in_features
            model.fc = nn.Linear(n_features, breakfast.N_CLASSES)
            self.model = model
            self.input_size = 224
        else:
            raise ValueError('no such model')
        self._load_saved_model()
        self.model = nn.DataParallel(self.model).cuda()

        self.loss_fn = nn.CrossEntropyLoss().cuda()
        if self.configs['optim'] == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        elif self.configs['optim'] == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.configs['momentum'],
                                       nesterov=self.configs['nesterov'], dampening=self.configs['dampening'],
                                       weight_decay=self.configs['weight-decay'])
        else:
            raise ValueError('no such optimizer')

        if self.configs['scheduler'] == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.configs['lr-step'],
                                                       gamma=self.configs['lr-decay'])
        elif self.configs['scheduler'] == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min',
                                                                  patience=self.configs['lr-step'])
        else:
            raise ValueError('no such scheduler')
        self._load_training_checkpoint()

    def train(self, train_data, test_data):
        train_segments, train_labels, train_logits = train_data
        test_segments, test_labels, test_logits = test_data

        train_dataset = TrainDataset(train_segments, train_labels, train_logits, input_size=self.input_size)
        test_dataset = TestDataset(test_segments, test_labels, test_logits, n_samples=self.n_test_segments,
                                   input_size=self.input_size)
        train_val_dataset = TestDataset(train_segments, train_labels, train_logits, input_size=self.input_size)

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
            self.scheduler.step(epoch)

    def train_step(self, train_dataset):
        print('INFO: training at epoch {}'.format(self.n_epochs))
        dataloader = tdata.DataLoader(train_dataset, shuffle=True, batch_size=self.train_batch_size, drop_last=True,
                                      collate_fn=train_dataset.collate_fn, pin_memory=True, num_workers=NUM_WORKERS)
        self.model.train()
        losses = []
        for feats, logits in tqdm(dataloader):
            feats = feats.cuda()
            logits = logits.cuda()

            self.optimizer.zero_grad()
            feats = self.model(feats)
            loss = self.loss_fn(feats, logits)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
        avg_loss = np.mean(losses)
        print('INFO: at epoch {0} loss = {1}'.format(self.n_epochs, avg_loss))
        self.tboard_writer.add_scalar('loss', avg_loss, self.n_epochs)

    def test_step(self, test_dataset):
        dataloader = tdata.DataLoader(test_dataset, shuffle=False, batch_size=self.test_batch_size,
                                      collate_fn=test_dataset.collate_fn, pin_memory=True, num_workers=NUM_WORKERS)
        self.model.eval()
        n_correct = 0
        n_predictions = 0
        with torch.no_grad():
            for feats, logits in tqdm(dataloader):
                bs, n_segments, n_channels, h, w = feats.shape
                feats = feats.cuda()
                logits = logits.cuda()

                feats = feats.view(-1, n_channels, h, w)
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

    def _load_saved_model(self, checkpoint_name='model'):
        checkpoint_file = os.path.join(self.checkpoint_dir, checkpoint_name + '.pth')
        if os.path.exists(checkpoint_file):
            print('INFO: loading checkpoint {}'.format(checkpoint_file))
            checkpoint = torch.load(checkpoint_file)
            self.model.load_state_dict(checkpoint['model'])
        else:
            print('INFO: checkpoint does not exist, continuing...')

    def _load_training_checkpoint(self, checkpoint_name='model'):
        checkpoint_file = os.path.join(self.checkpoint_dir, checkpoint_name + '.pth')
        if os.path.exists(checkpoint_file):
            print('INFO: loading checkpoint {}'.format(checkpoint_file))
            checkpoint = torch.load(checkpoint_file)
            self.optimizer.load_state_dict(checkpoint['optim'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            self.n_epochs = checkpoint['n-epochs']
        else:
            print('INFO: checkpoint does not exist, continuing...')

    def predict(self, prediction_segments):
        dataset = PredictionDataset(prediction_segments, self.n_test_segments)
        dataloader = tdata.DataLoader(dataset, shuffle=False, batch_size=self.test_batch_size, num_workers=NUM_WORKERS,
                                      pin_memory=True)
        self.model.eval()
        all_predictions = []
        with torch.no_grad():
            for feats in tqdm(dataloader):
                bs, n_segments, n_channels, h, w = feats.shape
                feats = feats.cuda()
                feats = feats.view(-1, n_channels, h, w)
                feats = self.model(feats)
                feats = feats.view(-1, self.n_test_segments, breakfast.N_CLASSES)
                feats = torch.sum(feats, dim=1)
                predictions = torch.argmax(feats, dim=1)
                predictions = predictions.detach().cpu().tolist()
                all_predictions.extend(predictions)
        return all_predictions


class TrainDataset(tdata.Dataset):
    def __init__(self, segments, segment_labels, segment_logits, input_size):
        super(TrainDataset, self).__init__()
        self.segments = segments
        self.segment_labels = segment_labels
        self.segment_logits = segment_logits
        self.transforms = transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __getitem__(self, idx):
        segment_dict = self.segments[idx]
        logit = self.segment_logits[idx]
        video_name = segment_dict['video-name']
        start, end = segment_dict['start'], segment_dict['end']
        assert start < end, '{0} has errors, logit {1}'.format(video_name, logit)

        frame_id = np.random.choice(np.arange(start, end))
        image = breakfast.read_frame(video_name, frame_id)
        image = self.transforms(image)
        return image, logit

    def __len__(self):
        return len(self.segments)

    @staticmethod
    def collate_fn(batch):
        images, logits = zip(*batch)
        images = torch.stack(images)
        logits = default_collate(logits)
        return images, logits


class TestDataset(TrainDataset):
    def __init__(self, segments, segment_labels, segment_logits, input_size, n_samples=25):
        super(TestDataset, self).__init__(segments, segment_labels, segment_logits, input_size)
        self.n_samples = n_samples
        self.transforms = transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __getitem__(self, idx):
        logit = self.segment_logits[idx]
        segment_frames = self.load_segment(idx)
        return segment_frames, logit

    def load_segment(self, idx):
        segment = self.segments[idx]
        video_name = segment['video-name']
        start, end = segment['start'], segment['end']

        sample_idxs = self._get_sample_idxs(start, end)
        segment_frames = [breakfast.read_frame(video_name, frame_id=frame_id) for frame_id in sample_idxs]
        segment_frames = [self.transforms(frame) for frame in segment_frames]
        segment_frames = torch.stack(segment_frames, dim=0)
        return segment_frames

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
        sample_idxs = sample_idxs.reshape(-1) + start
        return sample_idxs


class PredictionDataset(TestDataset):
    def __init__(self, segments, n_samples, input_size):
        super(PredictionDataset, self).__init__(segments=segments, segment_labels=None, segment_logits=None,
                                                n_samples=n_samples, input_size=input_size)

    def __getitem__(self, idx):
        selected = self.load_segment(idx)
        return selected


def get_clean_data(split):
    segments, labels, logits = breakfast.get_data(split)
    valid_segments = []
    valid_labels = []
    valid_logits = []
    for i, segment in enumerate(segments):
        if 48 > logits[i] > 0:  # remove walk in and walk out.
            valid_segments.append(segment)
            valid_labels.append(labels[i])
            valid_logits.append(logits[i])
    return [valid_segments, valid_labels, valid_logits]


def main():
    set_determinstic_mode()
    args = parse_args()
    trainer = Trainer(args.config)
    train_data = get_clean_data('train')
    test_data = get_clean_data('test')
    trainer.train(train_data, test_data)


if __name__ == '__main__':
    main()
