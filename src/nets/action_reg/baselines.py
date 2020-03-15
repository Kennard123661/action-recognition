import torch.nn as nn


class OneLayerMlp(nn.Module):
    def __init__(self, in_channels, n_classes, dropout=0.5):
        super(OneLayerMlp, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.net = nn.Sequential(nn.Dropout(dropout),
                                 nn.Linear(in_channels, n_classes, bias=False))

    def forward(self, inputs):
        out = self.net(inputs)
        return out


class ThreeLayerMlp(nn.Module):
    def __init__(self, in_channels, n_classes, base_channels=400, dropout=0.5):
        super(ThreeLayerMlp, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.base_channels = base_channels
        self.dropout = dropout
        self.net = nn.Sequential(nn.Linear(self.in_channels, self.base_channels, bias=False),
                                 nn.BatchNorm1d(self.base_channels),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(self.base_channels, self.base_channels, bias=False),
                                 nn.BatchNorm1d(self.base_channels),
                                 nn.ReLU(inplace=True),
                                 nn.Dropout(self.dropout),
                                 nn.Linear(self.base_channels, self.n_classes, bias=True))

    def forward(self, inputs):
        out = self.net(inputs)
        return out
