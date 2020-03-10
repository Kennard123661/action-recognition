import torch.nn as nn


class ClsHead(nn.Module):
    """Simplest classification head"""
    def __init__(self, dropout_ratio=0.8, in_channels=2048, num_classes=101, init_std=0.01, fcn_testing=False):

        super(ClsHead, self).__init__()

        self.dropout_ratio = float(dropout_ratio)
        self.in_channels = int(in_channels)
        self.init_std = float(init_std)
        self.num_classes = int(num_classes)
        self.dropout = nn.Dropout(p=self.dropout_ratio)
        self.fc_cls = nn.Linear(in_channels, num_classes)

    def init_weights(self):
        nn.init.normal_(self.fc_cls.weight, 0, self.init_std)
        nn.init.constant_(self.fc_cls.bias, 0)

    def forward(self, x):
        x = self.dropout(x)
        return self.fc_cls(x)
