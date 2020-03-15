import torch.nn as nn


class OneLayerMlp(nn.Module):
    def __init__(self, in_channels, embedding_size):
        super(OneLayerMlp, self).__init__()
        self.in_channels = in_channels
        self.embedding_size = embedding_size
        self.net = nn.Sequential(nn.Linear(in_channels, embedding_size, bias=False))

    def forward(self, inputs):
        out = self.net(inputs)
        return out


class ThreeLayerMlp(nn.Module):
    def __init__(self, in_channels, embedding_size, base_channels=400):
        super(ThreeLayerMlp, self).__init__()
        self.in_channels = in_channels
        self.embedding_size = embedding_size
        self.base_channels = base_channels
        self.net = nn.Sequential(nn.Linear(self.in_channels, self.base_channels, bias=False),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(self.base_channels, self.base_channels, bias=False),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(self.base_channels, self.embedding_size, bias=True))

    def forward(self, inputs):
        out = self.net(inputs)
        return out
