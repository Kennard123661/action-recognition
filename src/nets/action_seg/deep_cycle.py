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
        return (x + out) * mask[:, 0:1, :]


class SingleStageModel(nn.Module):
    def __init__(self, n_layers, base_channels, in_channels, out_channels):
        super(SingleStageModel, self).__init__()
        self.conv_1x1 = nn.Conv1d(in_channels, base_channels, 1)
        layers = [DilatedResidualLayer(2 ** i, base_channels, base_channels) for i in range(n_layers)]
        self.layers = nn.Sequential(*layers)
        self.conv_out = nn.Conv1d(base_channels, out_channels, 1)

    def forward(self, x, mask):
        out = self.conv_1x1(x)
        out = self.layers(out)
        out = self.conv_out(out) * mask[:, 0:1, :]
        return out


class VideoFeatGenerator(nn.Module):
    def __init__(self, n_layers, in_channels, reduction_factor=4):
        super(VideoFeatGenerator, self).__init__()
        self.stage1 = SingleStageModel(n_layers=n_layers, base_channels=in_channels // reduction_factor,
                                       in_channels=in_channels,
                                       out_channels=in_channels // reduction_factor)
        self.stage2 = SingleStageModel(n_layers=n_layers, base_channels=in_channels // reduction_factor,
                                       in_channels=in_channels // reduction_factor,
                                       out_channels=in_channels // reduction_factor)
        self.stage3 = SingleStageModel(n_layers=n_layers, base_channels=in_channels // reduction_factor,
                                       in_channels=in_channels // reduction_factor,
                                       out_channels=in_channels)
        self.stage4 = SingleStageModel(n_layers=n_layers, base_channels=in_channels,
                                       in_channels=in_channels, out_channels=in_channels)

    def forward(self, x, mask):
        out1 = self.stage1(x, mask)
        out2 = self.stage2(out1 * mask[:, 0:1, :], mask)
        out2 = out1 + out2
        out3 = self.stage3(out2 * mask[:, 0:1, :], mask)
        out3 = out3 + x
        out = self.stage4(out3 * mask[:, 0:1, :], mask)
        return out


class VideoFeatDiscriminator(nn.Module):
    def __init__(self, num_stages, num_layers, num_f_maps, dim):
        super(VideoFeatDiscriminator, self).__init__()
        self.stage1 = SingleStageModel(num_layers, num_f_maps, dim, 1)
        self.stages = nn.ModuleList([copy.deepcopy(SingleStageModel(num_layers, num_f_maps, 1, 1)) for s in range(num_stages-1)])

    def forward(self, x, mask):
        out = self.stage1(x, mask)
        outputs = out.unsqueeze(0)
        for s in self.stages:
            out = s(F.sigmoid(out) * mask[:, 0:1, :], mask)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        return outputs

