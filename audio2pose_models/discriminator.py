import torch
import torch.nn.functional as F
from torch import nn

class ConvNormRelu(nn.Module):
    def __init__(self, conv_type='1d', in_channels=3, out_channels=64, downsample=False,
                 kernel_size=None, stride=None, padding=None, norm='BN', leaky=False):
        super().__init__()
        if kernel_size is None:
            if downsample:
                kernel_size, stride, padding = 4, 2, 1
            else:
                kernel_size, stride, padding = 3, 1, 1

        if conv_type == '2d':
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            )
            if norm == 'BN':
                self.norm = nn.BatchNorm2d(out_channels)
            elif norm == 'IN':
                self.norm = nn.InstanceNorm2d(out_channels)
            else:
                raise NotImplementedError
        elif conv_type == '1d':
            self.conv = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            )
            if norm == 'BN':
                self.norm = nn.BatchNorm1d(out_channels)
            elif norm == 'IN':
                self.norm = nn.InstanceNorm1d(out_channels)
            else:
                raise NotImplementedError
        nn.init.kaiming_normal_(self.conv.weight)

        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=False) if leaky else nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if isinstance(self.norm, nn.InstanceNorm1d):
            x = self.norm(x.permute((0, 2, 1))).permute((0, 2, 1))  # normalize on [C]
        else:
            x = self.norm(x)
        x = self.act(x)
        return x


class PoseSequenceDiscriminator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        leaky = self.cfg.MODEL.DISCRIMINATOR.LEAKY_RELU

        self.seq = nn.Sequential(
            ConvNormRelu('1d', cfg.MODEL.DISCRIMINATOR.INPUT_CHANNELS, 256, downsample=True, leaky=leaky),  # B, 256, 64
            ConvNormRelu('1d', 256, 512, downsample=True, leaky=leaky),  # B, 512, 32
            ConvNormRelu('1d', 512, 1024, kernel_size=3, stride=1, padding=1, leaky=leaky),  # B, 1024, 16
            nn.Conv1d(1024, 1, kernel_size=3, stride=1, padding=1, bias=True)  # B, 1, 16
        )

    def forward(self, x):
        x = x.reshape(x.size(0), x.size(1), -1).transpose(1, 2)
        x = self.seq(x)
        x = x.squeeze(1)
        return x