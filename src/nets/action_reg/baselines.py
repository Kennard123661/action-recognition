import torch.nn as nn
import torch


class OneLayerMlp(nn.Module):
    def __init__(self, in_channels, n_classes, dropout=0.5):
        super(OneLayerMlp, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.net = nn.Sequential(nn.Dropout(dropout),
                                 nn.Linear(in_channels, n_classes, bias=False))

    def forward(self, inputs):
        batch_size, n_segments, _ = inputs.shape
        inputs = inputs.view(batch_size * n_segments, self.in_channels)
        out = self.net(inputs)
        out = out.view(batch_size, n_segments, self.n_classes)
        out = torch.sum(out, dim=1)  # sum over the number of segments
        return out
