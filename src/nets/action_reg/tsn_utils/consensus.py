import torch.nn as nn


class SimpleConsensus(nn.Module):
    def __init__(self, consensus_type='avg', dim=1):
        super(SimpleConsensus, self).__init__()

        assert consensus_type == 'avg'
        self.consensus_type = consensus_type
        self.dim = dim

    def forward(self, x):
        if self.consensus_type == 'avg':
            output = x.mean(dim=self.dim, keepdim=False)
        else:
            raise ValueError('no such simple consensus')
        return output
