import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, channels, x, y, planes=512):
        super(ConvBlock, self).__init__()
        self.channels, self.x, self.y = channels, x, y
        self.conv1 = nn.Conv2d(channels, planes, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(planes)

    def forward(self, s):
        s = s.view(-1, self.channels, self.x, self.y)  # batch_size x channels x board_x x board_y
        s = F.relu(self.bn1(self.conv1(s)))
        return s


class ResBlock(nn.Module):
    def __init__(self, inplanes=512, planes=512, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = F.relu(out)
        return out


class OutBlock(nn.Module):
    def __init__(self, channels, x, y, action_size, planes=512):
        super(OutBlock, self).__init__()
        self.channels, self.x, self.y, self.action_size = channels, x, y, action_size
        self.conv = nn.Conv2d(planes, channels, kernel_size=1)  # value head
        self.bn = nn.BatchNorm2d(channels)
        self.fc1 = nn.Linear(channels * x * y, 32)
        self.fc2 = nn.Linear(32, 1)

        self.conv1 = nn.Conv2d(planes, 32, kernel_size=1)  # policy head
        self.bn1 = nn.BatchNorm2d(32)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.fc = nn.Linear(x * y * 32, action_size)

    def forward(self, s):
        v = F.relu(self.bn(self.conv(s)))  # value head
        v = v.view(-1, self.channels * self.x * self.y)  # batch_size X channel X height X width
        v = F.relu(self.fc1(v))
        v = torch.tanh(self.fc2(v))

        p = F.relu(self.bn1(self.conv1(s)))  # policy head
        p = p.view(-1, self.x * self.y * 32)
        p = self.fc(p)
        p = self.logsoftmax(p).exp()
        return p, v


class NNetAlphaModule(nn.Module):
    def __init__(self, resblocks=5):
        super(NNetAlphaModule, self).__init__()
        channels, x, y, action_size = 5, 10, 15, 96
        self.resblocks = resblocks
        self.conv = ConvBlock(channels, x, y)
        for block in range(self.resblocks):
            setattr(self, "res_%i" % block, ResBlock())
        self.outblock = OutBlock(channels, x, y, action_size)

    def forward(self, s):
        s = self.conv(s)
        for block in range(self.resblocks):
            s = getattr(self, "res_%i" % block)(s)
        s = self.outblock(s)
        return s
