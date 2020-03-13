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
