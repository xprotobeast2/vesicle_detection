import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    def __init__(self, inc, outc, activation_constructor=nn.Tanh, batch_norm=True, pool_stride=2):
        super().__init__()
        net = []
        net.append(nn.Conv2d(inc, inc, 3, 1, 1))
        if activation_constructor is not None:
            net.append(activation_constructor())
        net.append(nn.Conv2d(inc, outc, 3, 1, 1))
        if activation_constructor is not None:
            net.append(activation_constructor())
        if batch_norm:
            net.append(nn.BatchNorm2d(outc))
        if pool_stride > 1:
            net.append(nn.MaxPool2d(pool_stride, return_indices=True))
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)


class ConvTBlock(nn.Module):
    def __init__(self, inc, outc, activation_constructor=nn.Tanh, batch_norm=True, pool_stride=2):
        super().__init__()

        net = []
        if pool_stride > 1:
            self.unpool = (nn.MaxUnpool2d(pool_stride))

        if batch_norm:
            net.append(nn.BatchNorm2d(inc))

        net.append(nn.ConvTranspose2d(inc, inc, 3, 1, 1))
        if activation_constructor is not None:
            net.append(activation_constructor())
        net.append(nn.Conv2d(inc, outc, 3, 1, 1))
        if activation_constructor is not None:
            net.append(activation_constructor())

        self.net = nn.Sequential(*net)

    def forward(self, fx, x, ind):
        return self.net(torch.cat([x, self.unpool(fx, ind)], dim=1))
