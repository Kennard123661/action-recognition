import os
import torch
import torch.nn as nn
from nets.action_reg.tsn_utils import backbone_nets
from nets.action_reg.tsn_utils import aggregate
from nets.action_reg.tsn_utils import consensus
from nets.action_reg.tsn_utils.cls_head import ClsHead
from nets.action_reg import ACTION_REG_PRETRAINED_DIR

TSN_PRETRAINED_DIR = os.path.join(ACTION_REG_PRETRAINED_DIR, 'tsn')
TSN_PRETRAINED_FLOW_DIR = os.path.join(TSN_PRETRAINED_DIR, 'flow')
TSN_PRETRAINED_RGB_DIR = os.path.join(TSN_PRETRAINED_DIR, 'rgb')

CKPT_FILENAMES = {
    'flow': {
        'ucf101': 'tsn_2d_flow_bninception_seg3_f1s1_b32_g8-151870b7.pth'
    },

    'rgb': {

    }
}

TSN_BACKBONES = {
    'flow': {
        'ucf101': 'bn-inception'
    },

    'rgb': {

    }
}

N_INPUT_CHANNELS = {
    'flow': {
        'ucf101': 10
    },

    'rgb': {

    }
}

AGGREGATION = {
    'flow': {
        'ucf101': aggregate.SimpleSpatialModule(spatial_type='avg', spatial_size=7)
    },

    'rgb': {

    }
}

CONSENSUS = {
    'flow': {
        'ucf101': consensus.SimpleConsensus(consensus_type='avg')
    },

    'rgb': {

    }
}

N_CLASSES = {
    'flow': {
        'ucf101': 101
    },

    'rgb': {}
}

DROPOUT_RATIO = {
    'flow': {
        'ucf101': 0.7
    },
    'rgb': {}
}


N_FC_CHANNELS = {
    'flow': {
        'ucf101': 1024
    },

    'rgb': {}
}


class TSN(nn.Module):
    def __init__(self, modality, ckpt_name, n_classes):
        super(TSN, self).__init__()
        self.n_classes = n_classes

        assert modality in ['rgb', 'flow']
        self.modality = modality

        backbone = TSN_BACKBONES[self.modality][ckpt_name]
        self.checkpoint_file = os.path.join(TSN_PRETRAINED_FLOW_DIR, CKPT_FILENAMES[self.modality][ckpt_name])
        self.fc_in_channels = N_FC_CHANNELS[self.modality][ckpt_name]
        self.n_input_channels = N_INPUT_CHANNELS[self.modality][ckpt_name]
        self.pool_fn = AGGREGATION[self.modality][ckpt_name]
        self.consensus = CONSENSUS[self.modality][ckpt_name]

        if backbone == 'bn-inception':
            self.backbone = backbone_nets.BNInception(self.n_input_channels)
        else:
            raise NotImplementedError

        self.cls_head = ClsHead(dropout_ratio=DROPOUT_RATIO[self.modality][ckpt_name],
                                in_channels=self.fc_in_channels,
                                num_classes=N_CLASSES[self.modality][ckpt_name])
        self.load_checkpoint()

        # change the number of classes
        if N_CLASSES[self.modality][ckpt_name] != self.n_classes:
            self.cls_head = ClsHead(dropout_ratio=DROPOUT_RATIO[self.modality][ckpt_name],
                                    in_channels=N_FC_CHANNELS[self.modality][ckpt_name],
                                    num_classes=self.n_classes)
            self.cls_head.init_weights()

    def load_checkpoint(self):
        model_state_dict = torch.load(self.checkpoint_file)['state_dict']
        self.load_state_dict(model_state_dict)

    def extract_feats(self, inputs):
        return self.backbone(inputs)

    def forward(self, inputs):
        batch_size, n_segments, _, height, width = inputs.shape
        print(self.n_input_channels)
        inputs = inputs.view(batch_size * n_segments, self.n_input_channels, height, width)
        inputs = self.extract_feats(inputs)
        inputs = self.pool_fn(inputs)
        inputs = inputs.view(batch_size, n_segments, self.fc_in_channels, height, width)
        inputs = self.consensus(inputs)
        cls_out = self.cls_head(inputs)
        return cls_out  # unnormalized classification score


def main():
    tsn = TSN(modality='flow', ckpt_name='ucf101', n_classes=101)
    import numpy as np
    dummy_inputs = torch.from_numpy(np.random.randn(3, 3, 10, 340, 256).astype(np.float32))
    cls_out = tsn(dummy_inputs)
    print(cls_out.shape)


if __name__ == '__main__':
    main()
