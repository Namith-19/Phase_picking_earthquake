import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=0, dilation=dilation)

    def forward(self, x):
        x = F.pad(x, (self.padding, 0))
        return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation):
        super().__init__()

        self.conv1 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.norm1 = nn.GroupNorm(1, channels)

        self.conv2 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.norm2 = nn.GroupNorm(1, channels)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):

        residual = x

        out = self.relu(self.norm1(self.conv1(x)))
        out = self.dropout(self.norm2(self.conv2(out)))

        out += residual

        return self.relu(out)


class ResidualCausalTCN(nn.Module):

    def __init__(self):

        super().__init__()

        self.input_conv = nn.Conv1d(3, 32, kernel_size=1)

        dilations = [1,2,4,8,16,32,64,128]

        self.blocks = nn.Sequential(
            *[ResidualBlock(32,5,d) for d in dilations]
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Linear(32,1)

    def forward(self, x):

        x = self.input_conv(x)

        x = self.blocks(x)

        x = self.pool(x).squeeze(-1)

        return self.fc(x)