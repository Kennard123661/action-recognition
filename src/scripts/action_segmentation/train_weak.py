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
import torch.nn.functional as F

R2PLU1D_MODELS = ['r2plus1d_34_32_ig65m', 'r2plus1d_34_32_kinetics', 'r2plus1d_34_8_ig65m', 'r2plus1d_34_8_kinetics']

if __name__ == '__main__':
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from scripts.action_segmentation import ACTION_SEG_CONFIG_DIR, ACTION_SEG_CHECKPOINT_DIR, ACTION_SEG_LOG_DIR
from scripts import set_determinstic_mode
from nets.action_seg import mstcn
import data.breakfast as breakfast
from scripts.action_segmentation.create_submission import get_cls_results

CHECKPOINT_DIR = os.path.join(ACTION_SEG_CHECKPOINT_DIR, 'weakly-supervised')
LOG_DIR = os.path.join(ACTION_SEG_LOG_DIR, 'weakly-supervised')
CONFIG_DIR = os.path.join(ACTION_SEG_CONFIG_DIR, 'weakly-supervised')
SUBMISSION_DIR = os.path.join('/mnt/HGST6/cs5242-project/submissions/action-segmentation/weakly-supervised-temp')
NUM_WORKERS = 2

N_STAGES = 4
N_LAYERS = 10
N_FEATURE_MAPS = 64
IN_CHANNELS = 2048

# todo change this.
SAMPLE_RATE = 1


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

        self.model = mstcn.MultiStageModel(num_stages=N_STAGES, num_layers=N_LAYERS, num_f_maps=N_FEATURE_MAPS,
                                           dim=IN_CHANNELS, num_classes=breakfast.N_MSTCN_CLASSES)
        self.model = self.model.cuda(self.device)
        self.coarse_cls_model = mstcn.SingleStageModel(num_layers=N_LAYERS, num_f_maps=N_FEATURE_MAPS,
                                                       dim=breakfast.N_MSTCN_CLASSES,
                                                       num_classes=len(breakfast.COARSE_LABELS)).cuda(self.device)
        self.coarse_ce_loss = nn.CrossEntropyLoss(ignore_index=-100).cuda(self.device)
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-100).cuda(self.device)
        self.mse_loss = nn.MSELoss(reduction='none').cuda(self.device)

        model_params = list(self.model.parameters()) + list(self.coarse_cls_model.parameters())
        if configs['optim'] == 'adam':
            self.optimizer = optim.Adam(model_params, lr=self.lr)
        elif configs['optim'] == 'sgd':
            self.optimizer = optim.SGD(model_params, lr=self.lr, momentum=configs['momentum'],
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
        self.submission_dir = os.path.join(SUBMISSION_DIR, self.experiment)

    def train(self, train_data, test_data):
        train_segments, train_labels, train_logits = train_data
        test_segments, test_labels, test_logits = test_data

        train_dataset = TrainDataset(train_segments, train_labels, train_logits, test_segments)
        test_dataset = TestDataset(test_segments, test_labels, test_logits)
        train_val_dataset = TestDataset(train_segments, train_labels, train_logits)

        start_epoch = self.n_epochs
        for epoch in range(start_epoch, self.max_epochs):
            self.n_epochs += 1
            self.train_step(train_dataset)
            self._save_checkpoint('model-{}'.format(self.n_epochs))
            self._save_checkpoint()  # update the latest model
            train_acc = self.test_step(train_val_dataset)
            test_acc = self.test_step(test_dataset)
            submission_acc = self.get_submission_acc(test_dataset)
            print('INFO: at epoch {}, the train accuracy is {} and the test accuracy is {} submission acc is {}'
                  .format(self.n_epochs, train_acc, test_acc, submission_acc))
            log_dict = {
                'train': train_acc,
                'test': test_acc,
                'submission': submission_acc
            }
            self.tboard_writer.add_scalars('accuracy', log_dict, self.n_epochs)

            if isinstance(self.scheduler, optim.lr_scheduler.StepLR):
                self.scheduler.step(epoch)

    def train_step(self, train_dataset):
        print('INFO: training at epoch {}'.format(self.n_epochs))
        dataloader = tdata.DataLoader(train_dataset, shuffle=True, batch_size=self.train_batch_size, drop_last=True,
                                      collate_fn=train_dataset.collate_fn, pin_memory=False, num_workers=NUM_WORKERS)
        self.model.train()
        self.coarse_cls_model.train()
        losses = []
        for source_features, source_logits, source_coarse_logits, source_masks, target_features, \
            target_coarse_logits, target_masks in tqdm(dataloader):
            source_features = source_features.cuda(self.device)
            source_logits = source_logits.cuda(self.device)
            source_coarse_logits = source_coarse_logits.cuda(self.device)
            source_masks = source_masks.cuda(self.device)
            target_features = target_features.cuda(self.device)
            target_coarse_logits = target_coarse_logits.cuda(self.device)
            target_masks = target_masks.cuda(self.device)
            self.optimizer.zero_grad()

            source_predictions = self.model(source_features, source_masks)
            fine_cls_loss = None
            for source_coarse_out in source_predictions:
                if fine_cls_loss is None:
                    fine_cls_loss = self.ce_loss(source_coarse_out.transpose(2, 1).contiguous()
                                                 .view(-1, breakfast.N_MSTCN_CLASSES),
                                                 source_logits.view(-1))
                else:
                    fine_cls_loss += self.ce_loss(source_coarse_out.transpose(2, 1).contiguous()
                                                  .view(-1, breakfast.N_MSTCN_CLASSES),
                                                  source_logits.view(-1))
                fine_cls_loss += 0.15 * torch.mean(torch.clamp(
                    self.mse_loss(F.log_softmax(source_coarse_out[:, :, 1:], dim=1),
                                  F.log_softmax(source_coarse_out.detach()[:, :, :-1], dim=1)), min=0, max=16) *
                                                   source_masks[:, :, 1:])

            coarse_predictions = []
            source_coarse_outs = self.coarse_cls_model(source_predictions[0], source_masks)
            for i, source_coarse_out in enumerate(source_coarse_outs):
                pred_length = torch.sum(source_masks[i, 0, :]).int()
                source_coarse_out = source_coarse_out[:, :pred_length]
                coarse_prediction = torch.mean(source_coarse_out, dim=1)
                coarse_predictions.append(coarse_prediction)

            target_predictions = self.model(target_features, target_masks)
            target_coarse_outs = self.coarse_cls_model(target_predictions[0], target_masks)
            for i, target_coarse_out in enumerate(target_coarse_outs):
                pred_length = torch.sum(target_masks[i, 0, :]).int()
                target_coarse_out = target_coarse_out[:, :pred_length]
                coarse_prediction = torch.mean(target_coarse_out, dim=1)
                coarse_predictions.append(coarse_prediction)
            coarse_predictions = torch.stack(coarse_predictions, dim=0)
            coarse_logits = torch.cat([source_coarse_logits, target_coarse_logits], dim=0)
            coarse_cls_loss = self.coarse_ce_loss(coarse_predictions, coarse_logits)

            total_loss = coarse_cls_loss + fine_cls_loss
            total_loss.backward()
            self.optimizer.step()
            losses.append(total_loss.item())
        avg_loss = np.mean(losses)
        print('INFO: at epoch {0} loss = {1}'.format(self.n_epochs, avg_loss))
        self.tboard_writer.add_scalar('loss', avg_loss, self.n_epochs)

        if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(avg_loss)

    def get_submission_acc(self, test_dataset):
        dataloader = tdata.DataLoader(test_dataset, shuffle=False, batch_size=self.test_batch_size,
                                      collate_fn=test_dataset.collate_fn, pin_memory=False, num_workers=NUM_WORKERS)
        self.model.eval()
        frame_level_predictions = []
        with torch.no_grad():
            for feats, logits, masks in tqdm(dataloader):
                feats = feats.cuda(self.device)
                masks = masks.cuda(self.device)
                predictions = self.model(feats, masks)
                predictions = torch.argmax(predictions[-1], dim=1)
                frame_level_predictions.extend(predictions.detach().cpu().numpy().tolist())
        return get_cls_results(frame_level_predictions, submission_dir=self.submission_dir)

    def test_step(self, test_dataset):
        dataloader = tdata.DataLoader(test_dataset, shuffle=False, batch_size=self.test_batch_size,
                                      collate_fn=test_dataset.collate_fn, pin_memory=False, num_workers=NUM_WORKERS)
        self.model.eval()
        n_correct = 0
        n_predictions = 0
        with torch.no_grad():
            for feats, logits, masks in tqdm(dataloader):
                feats = feats.cuda(self.device)
                logits = logits.cuda(self.device)
                masks = masks.cuda(self.device)
                predictions = self.model(feats, masks)
                predictions = torch.argmax(predictions[-1], dim=1)
                n_correct += ((predictions == logits).float() * masks[:, 0, :]).sum().item()
                n_predictions += torch.sum(masks[:, 0, :]).item()
        accuracy = n_correct / n_predictions
        return accuracy

    def _save_checkpoint(self, checkpoint_name='model'):
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        checkpoint_file = os.path.join(self.checkpoint_dir, checkpoint_name + '.pth')
        save_dict = {
            'model': self.model.state_dict(),
            'coarse-cls-model': self.coarse_cls_model.state_dict(),
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
            self.coarse_cls_model.load_state_dict(checkpoint['coarse-cls-model'])
            self.optimizer.load_state_dict(checkpoint['optim'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            self.n_epochs = checkpoint['n-epochs']
        else:
            print('INFO: checkpoint does not exist, continuing...')

    def predict(self, submission_feats):
        dataset = PredictionDataset(submission_feats)
        dataloader = tdata.DataLoader(dataset, shuffle=False, batch_size=self.test_batch_size,
                                      collate_fn=dataset.collate_fn, pin_memory=False, num_workers=NUM_WORKERS)
        with torch.no_grad():
            self.model.eval()
            submission_predictions = []
            for feats, masks in tqdm(dataloader):
                feats = feats.cuda(self.device)
                masks = masks.cuda(self.device)
                self.optimizer.zero_grad()
                predictions = self.model(feats, masks)
                predictions = torch.argmax(predictions[-1], dim=1)
                submission_predictions.extend(predictions.detach().cpu().numpy().tolist())
        return submission_predictions


class TrainDataset(tdata.Dataset):
    def __init__(self, source_feat_files, source_labels, source_logits, target_feat_files):
        super(TrainDataset, self).__init__()
        self.source_feat_files = source_feat_files
        self.source_labels = source_labels
        self.source_logits = source_logits
        self.target_feat_files = target_feat_files
        self.n_targets = len(target_feat_files)
        self.n_classes = breakfast.N_MSTCN_CLASSES

    def __len__(self):
        return len(self.source_feat_files)

    def __getitem__(self, idx):
        source_feat_file = self.source_feat_files[idx]
        source_coarse_label = os.path.split(source_feat_file)[-1]
        source_coarse_label = source_coarse_label.split('.')[0].split('_')[-1]
        source_features = np.load(source_feat_file)

        target_idx = np.random.choice(self.n_targets)
        target_feat_file = self.target_feat_files[target_idx]
        target_coarse_label = os.path.split(target_feat_file)[-1]
        target_coarse_label = target_coarse_label.split('.')[0].split('_')[-1]
        target_features = np.load(target_feat_file)

        source_logits = self.source_logits[idx]
        assert source_features.shape[1] == len(source_logits)
        source_features = source_features[:, ::SAMPLE_RATE]
        source_logits = np.array(source_logits)[::SAMPLE_RATE]

        source_features = torch.from_numpy(source_features)
        target_features = torch.from_numpy(target_features)
        source_logits = torch.from_numpy(source_logits)
        source_coarse_logit = breakfast.COARSE_LABELS.index(source_coarse_label)
        target_coarse_logit = breakfast.COARSE_LABELS.index(target_coarse_label)
        return source_features, source_logits, source_coarse_logit, target_features, target_coarse_logit

    @staticmethod
    def _pad_features(features, coarse_logits, logits=None):
        max_video_length = 0
        for feature in features:
            max_video_length = max(feature.shape[1], max_video_length)

        padded_features = torch.zeros(len(features), IN_CHANNELS, max_video_length, dtype=torch.float)
        padded_logits = torch.zeros(len(features), max_video_length, dtype=torch.long) * -100
        masks = torch.zeros(len(features), breakfast.N_MSTCN_CLASSES, max_video_length, dtype=torch.float)

        coarse_logits = torch.from_numpy(np.array(coarse_logits))
        for i, feature in enumerate(features):
            video_len = feature.shape[1]
            padded_features[i, :, :video_len] = feature
            if logits is not None:
                padded_logits[i, :video_len] = logits[i]
            masks[i, :, :video_len] = torch.ones(breakfast.N_MSTCN_CLASSES, video_len)
        if logits is None:
            return padded_features, coarse_logits, masks
        else:
            return padded_features, padded_logits, coarse_logits, masks

    @staticmethod
    def collate_fn(batch):
        source_features, source_logits, source_coarse_logits, target_features, target_coarse_logits = zip(*batch)
        source_features, source_logits, source_coarse_logits, source_masks = \
            TrainDataset._pad_features(features=source_features, logits=source_logits,
                                       coarse_logits=source_coarse_logits)
        target_features, target_coarse_logits, target_masks = \
            TrainDataset._pad_features(features=target_features, coarse_logits=target_coarse_logits)
        return source_features, source_logits, source_coarse_logits, source_masks, target_features, \
               target_coarse_logits, target_masks


class TestDataset(TrainDataset):
    def __init__(self, feat_files, labels, logits):
        super(TrainDataset, self).__init__()
        self.video_feat_files = feat_files
        self.labels = labels
        self.logits = logits
        self.n_classes = breakfast.N_MSTCN_CLASSES

    def __len__(self):
        return len(self.video_feat_files)

    def __getitem__(self, idx):
        video_feat_file = self.video_feat_files[idx]
        features = np.load(video_feat_file)
        logits = self.logits[idx]
        assert features.shape[1] == len(logits)
        features = features[:, ::SAMPLE_RATE]
        logits = np.array(logits)[::SAMPLE_RATE]

        features = torch.from_numpy(features)
        logits = torch.from_numpy(logits)
        return features, logits

    @staticmethod
    def collate_fn(batch):
        features, logits = zip(*batch)
        max_video_length = 0
        for feature in features:
            max_video_length = max(feature.shape[1], max_video_length)

        padded_features = torch.zeros(len(features), IN_CHANNELS, max_video_length, dtype=torch.float)
        padded_logits = torch.zeros(len(features), max_video_length, dtype=torch.long) * -100
        masks = torch.zeros(len(features), breakfast.N_MSTCN_CLASSES, max_video_length, dtype=torch.float)

        for i, feature in enumerate(features):
            video_len = feature.shape[1]
            padded_features[i, :, :video_len] = feature
            padded_logits[i, :video_len] = logits[i]
            masks[i, :, :video_len] = torch.ones(breakfast.N_MSTCN_CLASSES, video_len)
        return padded_features, padded_logits, masks


class PredictionDataset(tdata.Dataset):
    def __init__(self, feat_files):
        super(PredictionDataset, self).__init__()
        self.video_feat_files = feat_files
        self.n_classes = breakfast.N_MSTCN_CLASSES

    def __len__(self):
        return len(self.video_feat_files)

    def __getitem__(self, idx):
        video_feat_file = self.video_feat_files[idx]
        features = np.load(video_feat_file)
        features = features[:, ::SAMPLE_RATE]
        features = torch.from_numpy(features)
        return features

    @staticmethod
    def collate_fn(batch):
        features = batch
        max_video_length = 0
        for feature in features:
            max_video_length = max(feature.shape[1], max_video_length)

        padded_features = torch.zeros(len(features), IN_CHANNELS, max_video_length, dtype=torch.float)
        masks = torch.zeros(len(features), breakfast.N_MSTCN_CLASSES, max_video_length, dtype=torch.float)

        for i, feature in enumerate(features):
            video_len = feature.shape[1]
            padded_features[i, :, :video_len] = feature
            masks[i, :, :video_len] = torch.ones(breakfast.N_MSTCN_CLASSES, video_len)
        return padded_features, masks


def _parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-c', '--config', required=True, type=str, help='config filename e.g -c base')
    argparser.add_argument('-d', '--device', default=0, choices=np.arange(torch.cuda.device_count()),
                           type=int, help='device to run on')
    return argparser.parse_args()


def main():
    set_determinstic_mode(seed=1538574472)
    args = _parse_args()
    trainer = Trainer(args.config, args.device)
    train_data = breakfast.get_mstcn_data('train')
    test_data = breakfast.get_mstcn_data('test')
    trainer.train(train_data, test_data)


if __name__ == '__main__':
    main()
    # print(torch.hub.list("moabitcoin/ig65m-pytorch"))
    # model = torch.hub.load("moabitcoin/ig65m-pytorch", "r2plus1d_34_32_ig65m", num_classes=359, pretrained=True)
