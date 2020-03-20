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
from nets.action_seg import mstcn, cyclegan
import data.breakfast as breakfast

MSTCN_CHECKPOINT_DIR = os.path.join(ACTION_SEG_CHECKPOINT_DIR, 'mstcn')
CHECKPOINT_DIR = os.path.join(ACTION_SEG_CHECKPOINT_DIR, 'cyclegan')
LOG_DIR = os.path.join(ACTION_SEG_LOG_DIR, 'cyclegan')
CONFIG_DIR = os.path.join(ACTION_SEG_CONFIG_DIR, 'cyclegan')
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

        self.gen_lr = configs['gen-lr']
        self.dis_lr = configs['dis-lr']
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

        self.source_to_target_gen = cyclegan.VideoFeatGenerator(n_layers=10, in_channels=IN_CHANNELS,
                                                                reduction_factor=4).cuda(self.device)
        self.target_to_source_gen = cyclegan.VideoFeatGenerator(n_layers=10, in_channels=IN_CHANNELS,
                                                                reduction_factor=4).cuda(self.device)
        self.source_dis = cyclegan.VideoFeatDiscriminator(num_layers=N_LAYERS, num_f_maps=N_FEATURE_MAPS,
                                                          dim=IN_CHANNELS).cuda(self.device)
        self.target_dis = cyclegan.VideoFeatDiscriminator(num_layers=N_LAYERS, num_f_maps=N_FEATURE_MAPS,
                                                          dim=IN_CHANNELS).cuda(self.device)

        self.mstcn_model = mstcn.MultiStageModel(num_stages=N_STAGES, num_layers=N_LAYERS,
                                                 num_f_maps=N_FEATURE_MAPS,
                                                 dim=IN_CHANNELS, num_classes=breakfast.N_MSTCN_CLASSES).cpu()
        self.mstcn_model.eval()
        self.mstcn_model_config = configs['mstcn-config']
        self._load_mstcn_model()

        self.reconstruction_loss_fn = nn.L1Loss()

        gen_parameters = list(self.target_to_source_gen.parameters()) + list(self.source_to_target_gen.parameters())
        dis_parameters = list(self.target_dis.parameters()) + list(self.source_dis.parameters())
        if configs['optim'] == 'adam':
            self.gen_optimizer = optim.Adam(gen_parameters, lr=self.gen_lr)
            self.dis_optimizer = optim.Adam(dis_parameters, lr=self.dis_lr)
        elif configs['optim'] == 'sgd':
            self.gen_optimizer = optim.SGD(gen_parameters, lr=self.gen_lr, momentum=configs['momentum'],
                                           nesterov=configs['nesterov'])
            self.dis_optimizer = optim.SGD(dis_parameters, lr=self.gen_lr, momentum=configs['momentum'],
                                           nesterov=configs['nesterov'])
        else:
            raise ValueError('no such optimizer')

        if configs['scheduler'] == 'step':
            self.gen_scheduler = optim.lr_scheduler.StepLR(self.gen_optimizer, step_size=configs['gen-lr-step'],
                                                           gamma=configs['gen-lr-decay'])
            self.dis_scheduler = optim.lr_scheduler.StepLR(self.dis_optimizer, step_size=configs['dis-lr-step'],
                                                           gamma=configs['dis-lr-decay'])

        elif configs['scheduler'] == 'plateau':
            self.gen_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.gen_optimizer, mode='min',
                                                                      patience=configs['gen-lr-step'])
            self.dis_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.dis_optimizer, mode='min',
                                                                      patience=configs['dis-lr-step'])
        else:
            raise ValueError('no such scheduler')
        self._load_checkpoint()

    def train(self, train_data, test_data):
        train_segments, train_labels, train_logits = train_data
        test_segments, test_labels, test_logits = test_data

        train_dataset = TrainDataset(train_segments, test_segments)
        test_dataset = TestDataset(test_segments, test_labels, test_logits)

        start_epoch = self.n_epochs
        for epoch in range(start_epoch, self.max_epochs):
            self.n_epochs += 1
            self.train_step(train_dataset)
            self._save_checkpoint('model-{}'.format(self.n_epochs))
            self._save_checkpoint()  # update the latest model
            test_acc = self.test_step(test_dataset)
            print('INFO: at epoch {}, the test accuracy is {}'.format(self.n_epochs, test_acc))
            log_dict = {
                'test': test_acc
            }
            self.tboard_writer.add_scalars('accuracy', log_dict, self.n_epochs)

            if isinstance(self.dis_scheduler, optim.lr_scheduler.StepLR):
                self.dis_scheduler.step(epoch)
            if isinstance(self.gen_scheduler, optim.lr_scheduler.StepLR):
                self.gen_scheduler.step(epoch)

    def _load_mstcn_model(self):
        checkpoint_file = os.path.join(MSTCN_CHECKPOINT_DIR, self.mstcn_model_config, 'model.pth')
        assert os.path.exists(checkpoint_file)
        print('INFO: loading pretrained mstcn checkpoint {}'.format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file, map_location='cpu')
        self.mstcn_model.load_state_dict(checkpoint['model'])

    def _get_reconstruction_loss(self, original_feats, reconstructed_feats, masks):
        reconstruction_loss = None
        for i in range(len(reconstructed_feats)):
            reconstructed_source = reconstructed_feats[i]  # D x N
            original_source = original_feats[i]  # D x N

            mask = masks[i]
            video_len = torch.sum(mask).int().item()
            reconstructed_source = reconstructed_source[:video_len]
            original_source = original_source[:video_len]
            if reconstruction_loss is None:
                reconstruction_loss = self.reconstruction_loss_fn(reconstructed_source.transpose(1, 0),
                                                                  original_source.transpose(1, 0))
            else:
                reconstruction_loss += self.reconstruction_loss_fn(reconstructed_source.transpose(1, 0),
                                                                   original_source.transpose(1, 0))
        return reconstruction_loss / len(reconstructed_feats)

    @staticmethod
    def _get_disciminator_loss(dis_model, positive_feats, negative_feats, positive_masks, negative_masks):
        positive_logits = dis_model(positive_feats, positive_masks)
        negative_logits = dis_model(negative_feats, negative_masks)

        loss = None
        for i, pos_logits in enumerate(positive_logits):
            pos_mask = positive_masks[i]
            n_positive = torch.sum(pos_mask).int().item()
            for j, neg_logits in enumerate(negative_logits):
                neg_mask = negative_masks[j]
                n_negative = torch.sum(neg_mask).int().item()
                pos_logits = pos_logits[:, :n_positive]
                neg_logits = neg_logits[:, :n_negative]
                if loss is None:
                    loss = (torch.mean(pos_logits) - torch.mean(neg_logits) - 1)**2
                else:
                    loss += (torch.mean(pos_logits) - torch.mean(neg_logits) - 1)**2
        n_losses = positive_logits.shape[0] * negative_logits.shape[0]
        dis_loss = loss / n_losses
        return dis_loss

    def train_step(self, train_dataset):
        print('INFO: training at epoch {}'.format(self.n_epochs))
        dataloader = tdata.DataLoader(train_dataset, shuffle=True, batch_size=self.train_batch_size, drop_last=True,
                                      collate_fn=train_dataset.collate_fn, pin_memory=False, num_workers=NUM_WORKERS)

        self.source_to_target_gen.train()
        self.target_to_source_gen.train()
        self.source_dis.train()
        self.target_dis.train()
        dis_losses = []
        gen_losses = []
        for source_feats, source_masks, target_feats, target_masks in tqdm(dataloader):
            source_feats = source_feats.cuda(self.device)
            source_masks = source_masks.cuda(self.device)
            target_feats = target_feats.cuda(self.device)
            target_masks = target_masks.cuda(self.device)
            self.gen_optimizer.zero_grad()
            self.dis_optimizer.zero_grad()

            # train the generator
            fake_target_feats = self.source_to_target_gen(source_feats, source_masks)
            fake_source_feats = self.target_to_source_gen(target_feats, target_masks)
            reconstructed_source_feats = self.target_to_source_gen(fake_target_feats, source_masks)  # B x D x N
            reconstructed_target_feats = self.source_to_target_gen(fake_source_feats, target_masks)
            source_reconstruction_loss = self._get_reconstruction_loss(source_feats, reconstructed_source_feats,
                                                                       source_masks)
            target_reconstruction_loss = self._get_reconstruction_loss(target_feats, reconstructed_target_feats,
                                                                       target_masks)
            source_gen_loss = self._get_disciminator_loss(dis_model=self.source_dis, positive_feats=fake_source_feats,
                                                          negative_feats=source_feats, positive_masks=target_masks,
                                                          negative_masks=source_masks)
            target_gen_loss = self._get_disciminator_loss(dis_model=self.target_dis, positive_feats=fake_target_feats,
                                                          negative_feats=target_feats, positive_masks=source_masks,
                                                          negative_masks=target_masks)
            gen_loss = source_reconstruction_loss + target_reconstruction_loss + source_gen_loss + target_gen_loss
            gen_loss.backward()
            self.gen_optimizer.step()

            # train the discriminator
            self.dis_optimizer.zero_grad()
            source_dis_loss = self._get_disciminator_loss(self.source_dis, positive_feats=source_feats,
                                                          negative_feats=fake_source_feats.detach(),
                                                          positive_masks=source_masks, negative_masks=target_masks)
            target_dis_loss = self._get_disciminator_loss(self.target_dis, positive_feats=target_feats,
                                                          negative_feats=fake_target_feats.detach(),
                                                          positive_masks=target_masks,
                                                          negative_masks=source_masks)
            dis_loss = (target_dis_loss + source_dis_loss) / 2
            dis_loss.backward()
            self.dis_optimizer.step()

            dis_losses.append(dis_loss.item())
            gen_losses.append(gen_loss.item())
        avg_dis_loss = np.mean(dis_losses)
        avg_gen_loss = np.mean(gen_losses)
        print('INFO: at epoch {0} gen-loss = {1}, dis-loss = {2}'.format(self.n_epochs, avg_gen_loss, avg_dis_loss))
        loss_dict = {
            'dis-loss': avg_dis_loss,
            'gen-loss': avg_gen_loss
        }
        self.tboard_writer.add_scalars('losses', loss_dict, self.n_epochs)

        if isinstance(self.dis_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            self.dis_scheduler.step(avg_dis_loss)
        if isinstance(self.gen_optimizer, optim.lr_scheduler.ReduceLROnPlateau):
            self.gen_scheduler.step(avg_gen_loss)

    def test_step(self, test_dataset):
        dataloader = tdata.DataLoader(test_dataset, shuffle=False, batch_size=self.test_batch_size,
                                      collate_fn=test_dataset.collate_fn, pin_memory=False, num_workers=NUM_WORKERS)
        self.mstcn_model.eval()
        self.target_to_source_gen.eval()
        self.mstcn_model = self.mstcn_model.cuda(self.device)
        n_correct = 0
        n_predictions = 0
        with torch.no_grad():
            for feats, logits, masks in tqdm(dataloader):
                feats = feats.cuda(self.device)
                logits = logits.cuda(self.device)
                masks = masks.cuda(self.device)

                generated_feats = self.target_to_source_gen(feats, masks[:, 0:1, :])
                predictions = self.mstcn_model(generated_feats, masks)
                predictions = torch.argmax(predictions[-1], dim=1)
                n_correct += ((predictions == logits).float() * masks[:, 0, :]).sum().item()
                n_predictions += torch.sum(masks[:, 0, :]).item()
        accuracy = n_correct / n_predictions
        self.mstcn_model = self.mstcn_model.cpu()
        return accuracy

    def _save_checkpoint(self, checkpoint_name='model'):
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        checkpoint_file = os.path.join(self.checkpoint_dir, checkpoint_name + '.pth')
        save_dict = {
            'source-target-gen': self.source_to_target_gen.state_dict(),
            'target-source-gen': self.target_to_source_gen.state_dict(),
            'source-dis': self.source_dis.state_dict(),
            'target-dis': self.target_dis.state_dict(),
            'gen-optim': self.gen_optimizer.state_dict(),
            'dis-optim': self.dis_optimizer.state_dict(),
            'gen-scheduler': self.gen_scheduler.state_dict(),
            'dis-scheduler': self.dis_scheduler.state_dict(),
            'n-epochs': self.n_epochs
        }
        torch.save(save_dict, checkpoint_file)

    def _load_checkpoint(self, checkpoint_name='model'):
        checkpoint_file = os.path.join(self.checkpoint_dir, checkpoint_name + '.pth')
        if os.path.exists(checkpoint_file):
            print('INFO: loading checkpoint {}'.format(checkpoint_file))
            checkpoint = torch.load(checkpoint_file)
            self.source_to_target_gen.load_state_dict(checkpoint['source-target-gen'])
            self.target_to_source_gen.load_state_dict(checkpoint['target-source-gen'])
            self.source_dis.load_state_dict(checkpoint['source-dis'])
            self.target_dis.load_state_dict(checkpoint['target-dis'])
            self.gen_optimizer.load_state_dict(checkpoint['gen-optim'])
            self.dis_optimizer.load_state_dict(checkpoint['dis-optim'])
            self.gen_scheduler.load_state_dict(checkpoint['gen-scheduler'])
            self.dis_scheduler.load_state_dict(checkpoint['dis-scheduler'])
            self.n_epochs = checkpoint['n-epochs']
        else:
            print('INFO: checkpoint does not exist, continuing...')

    def predict(self, submission_feats):
        dataset = PredictionDataset(submission_feats)
        dataloader = tdata.DataLoader(dataset, shuffle=False, batch_size=self.test_batch_size,
                                      collate_fn=dataset.collate_fn, pin_memory=False, num_workers=NUM_WORKERS)

        submission_predictions = []
        for feats, masks in tqdm(dataloader):
            feats = feats.cuda(self.device)
            masks = masks.cuda(self.device)
            predictions = self.mstcn_model(feats, masks)
            predictions = torch.argmax(predictions[-1], dim=1)
            for i, prediction in enumerate(predictions):
                prediction_len = torch.sum(masks[i, 0, :].int()).item()
                prediction = prediction[:prediction_len].detach().cpu().numpy().tolist()
                submission_predictions.append(prediction)
        return submission_predictions


class TrainDataset(tdata.Dataset):
    def __init__(self, source_feat_files, target_feat_files):
        super(TrainDataset, self).__init__()
        self.source_feat_files = source_feat_files
        self.target_feat_files = target_feat_files
        self.n_classes = breakfast.N_MSTCN_CLASSES

    def __len__(self):
        return max(len(self.source_feat_files), len(self.target_feat_files))

    def __getitem__(self, idx):
        n_source = len(self.source_feat_files)
        n_target = len(self.target_feat_files)

        source_idx = np.random.choice(np.arange(n_source))
        target_idx = np.random.choice(np.arange(n_target))

        source_feat_file = self.source_feat_files[source_idx]
        target_feat_file = self.source_feat_files[target_idx]

        source_feats = np.load(source_feat_file)
        target_feats = np.load(target_feat_file)

        source_feats = torch.from_numpy(source_feats)
        target_feats = torch.from_numpy(target_feats)
        return source_feats, target_feats

    @staticmethod
    def _pad_features(features):
        max_video_length = 0
        for feature in features:
            max_video_length = max(feature.shape[1], max_video_length)

        padded_features = torch.zeros(len(features), IN_CHANNELS, max_video_length, dtype=torch.float)
        masks = torch.zeros(len(features), max_video_length, dtype=torch.float)

        for i, feature in enumerate(features):
            video_len = feature.shape[1]
            padded_features[i, :, :video_len] = feature
            masks[i, :video_len] = torch.ones(video_len)
        return padded_features, masks

    @staticmethod
    def collate_fn(batch):
        source_features, target_features = zip(*batch)
        source_features, source_masks = TrainDataset._pad_features(source_features)
        target_features, target_masks = TrainDataset._pad_features(target_features)
        return source_features, source_masks, target_features, target_masks


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
