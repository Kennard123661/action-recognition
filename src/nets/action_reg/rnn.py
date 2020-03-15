import torch.nn as nn
import torch
from torch.nn.utils.rnn import pad_sequence


class Baseline(nn.Module):
    def __init__(self, n_inputs, n_classes, hidden_size, aggregate):
        super(Baseline, self).__init__()
        self.n_inputs = int(n_inputs)
        self.n_classes = int(n_classes)
        self.hidden_size = int(hidden_size)
        self.aggregate = str(aggregate)
        assert self.aggregate in ['last', 'mean']

        self.rnn = nn.LSTMCell(self.n_inputs, hidden_size=self.hidden_size, bias=True)
        self.c0_net = nn.Sequential(
            nn.Linear(in_features=self.n_inputs, out_features=self.n_inputs, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=self.n_inputs, out_features=self.hidden_size, bias=False)
        )
        self.h0_net = nn.Sequential(
            nn.Linear(in_features=self.n_inputs, out_features=self.n_inputs, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=self.n_inputs, out_features=self.hidden_size, bias=False)
        )

        self.fc = nn.Sequential(nn.Linear(self.hidden_size, self.n_classes, bias=True))

    def _get_init_states(self, input_means):
        h0 = self.h0_net(input_means)
        c0 = self.c0_net(input_means)
        return h0, c0

    def forward(self, input_features, segment_lengths):
        """
        :param input_features: should be BT x C
        :param segment_lengths: B vector of lengths
        :return:
        """
        input_means = []
        segment_inputs = []
        start = 0
        for segment_len in segment_lengths:
            end = start + segment_len
            segment_features = input_features[start:end]
            segment_inputs.append(segment_features)
            input_means.append(torch.sum(segment_features, dim=0))
            start = end
        input_means = torch.stack(input_means, dim=0)
        segment_inputs = pad_sequence(sequences=segment_inputs, batch_first=True, padding_value=0)
        # print(segment_inputs.shape)
        max_segment_length = segment_inputs.shape[1]

        ht, ct = self._get_init_states(input_means=input_means)
        h_outs = []
        for t in range(max_segment_length):
            ht, ct = self.rnn(segment_inputs[:, t], (ht, ct))
            h_outs.append(ht)
        h_outs = torch.stack(h_outs, dim=0)

        fc_inputs = []
        for i, segment_len in enumerate(segment_lengths):
            h_out = h_outs[i, :segment_len]
            if self.aggregate == 'last':
                fc_input = h_out[-1]
            elif self.aggregate == 'mean':
                fc_input = torch.mean(h_out, dim=0)
            else:
                raise ValueError('no such aggregation')
            fc_inputs.append(fc_input)
        fc_inputs = torch.stack(fc_inputs, dim=0)
        return self.fc(fc_inputs)




