import sys

import torch
import torch.nn.functional as F
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, channels, x, y, features):
        super(ConvBlock, self).__init__()
        self.channels, self.x, self.y = channels, x, y
        self.conv1 = nn.Conv2d(channels, features, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(features)

    def forward(self, s):
        s = s.view(-1, self.channels, self.x, self.y)  # batch_size x channels x board_x x board_y
        s = F.relu(self.bn1(self.conv1(s)))
        return s


class ResBlock(nn.Module):
    def __init__(self, features, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(features)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(features)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = F.relu(out)
        return out


class ValueHeadBlock(nn.Module):
    def __init__(self, channels, x, y, features):
        super(ValueHeadBlock, self).__init__()
        self.channels, self.x, self.y, = channels, x, y
        self.conv = nn.Conv2d(features, channels, kernel_size=1)  # value head
        self.bn = nn.BatchNorm2d(channels)
        self.fc1 = nn.Linear(channels * x * y, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, s):
        v = F.relu(self.bn(self.conv(s)))  # value head
        v = v.view(-1, self.channels * self.x * self.y)  # batch_size X channel X height X width
        v = F.relu(self.fc1(v))
        v = torch.tanh(self.fc2(v))

        return v


class PolicyHeadBlock(nn.Module):
    def __init__(self, x, y, action_size, features):
        super(PolicyHeadBlock, self).__init__()
        self.x, self.y = x, y
        self.conv = nn.Conv2d(features, 32, kernel_size=1)  # policy head
        self.bn = nn.BatchNorm2d(32)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.fc = nn.Linear(x * y * 32, action_size)

    def forward(self, s):
        p = F.relu(self.bn(self.conv(s)))  # policy head
        p = p.view(-1, self.x * self.y * 32)
        p = self.fc(p)
        p = self.logsoftmax(p).exp()
        return p


class NNetAlphaModule(nn.Module):
    def __init__(self, resblocks=5, features=128):
        super(NNetAlphaModule, self).__init__()
        channels, x, y, action_size = 5, 10, 15, 96
        self.resblocks = resblocks
        self.conv = ConvBlock(channels, x, y, features)
        for block in range(self.resblocks):
            setattr(self, "res_%i" % block, ResBlock(features))
        self.valuehead = ValueHeadBlock(channels, x, y, features)
        self.policyhead = PolicyHeadBlock(x, y, action_size, features)

    def forward(self, s):
        s = self.conv(s)
        for block in range(self.resblocks):
            s = getattr(self, "res_%i" % block)(s)
        p = self.policyhead(s)
        v = self.valuehead(s)
        return p, v


class AlphaLoss(torch.nn.Module):
    def __init__(self, loss_function="MSE"):
        super(AlphaLoss, self).__init__()
        self.loss_function = loss_function

    def forward(self, v_out, v, pv_out, pv):
        if self.loss_function == "MSE":
            value_error = (v - v_out) ** 2
        else:
            value_error = abs((v - v_out))
        policy_error = torch.sum((-pv * (1e-8 + pv_out.float()).float().log()), 1)
        total_error = (value_error.view(-1).float() + policy_error).mean()
        return total_error, value_error.view(-1).float().mean(), policy_error.mean()
