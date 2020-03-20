import torch.nn as nn
import torch.nn.functional as F
import copy
import torch


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x, mask):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out) * mask


class SingleStageModel(nn.Module):
    def __init__(self, n_layers, base_channels, in_channels, out_channels):
        super(SingleStageModel, self).__init__()
        self.conv_1x1 = nn.Conv1d(in_channels, base_channels, 1)
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(2 ** i, base_channels, base_channels))
                                     for i in range(n_layers)])

        self.conv_out = nn.Conv1d(base_channels, out_channels, 1)

    def forward(self, x, mask):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out, mask)
        out = self.conv_out(out) * mask
        return out


class VideoFeatGenerator(nn.Module):
    def __init__(self, n_layers, in_channels, reduction_factor=4):
        super(VideoFeatGenerator, self).__init__()
        # self.stage1 = SingleStageModel(n_layers=n_layers, base_channels=in_channels // reduction_factor,
        #                                in_channels=in_channels,
        #                                out_channels=in_channels // reduction_factor)
        # self.stage2 = SingleStageModel(n_layers=n_layers, base_channels=in_channels // reduction_factor,
        #                                in_channels=in_channels // reduction_factor,
        #                                out_channels=in_channels // reduction_factor)
        # self.stage3 = SingleStageModel(n_layers=n_layers, base_channels=in_channels // reduction_factor,
        #                                in_channels=in_channels // reduction_factor,
        #                                out_channels=in_channels)
        # self.stage4 = SingleStageModel(n_layers=n_layers, base_channels=in_channels,
        #                                in_channels=in_channels, out_channels=in_channels)
        self.net = SingleStageModel(n_layers=n_layers, base_channels=in_channels // reduction_factor,
                                    in_channels=in_channels, out_channels=in_channels)

    def forward(self, x, mask):
        # out1 = self.stage1(x, mask)
        # out2 = self.stage2(out1 * mask, mask)
        # out2 = out1 + out2
        # out3 = self.stage3(out2 * mask, mask)
        # out3 = out3 + x
        # out = self.stage4(out3 * mask, mask)
        out = self.net(x, mask)
        return out


class VideoFeatDiscriminator(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim):
        super(VideoFeatDiscriminator, self).__init__()
        self.net = SingleStageModel(num_layers, num_f_maps, dim, 1)

    def forward(self, x, mask):
        out = self.net(x, mask)
        out = torch.sigmoid(out)
        return out

