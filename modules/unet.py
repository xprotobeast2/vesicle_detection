import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class UNet128(nn.Module):
    # Courtesy code I wrote for center_pivot_detection project
    def __init__(self, in_channels=1, n_classes=2, do_log_softmax=False):
        super().__init__()
        self.do_log_softmax = do_log_softmax
        # c,256x256 -> 256x256
        # Level 0
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels, 8, 3, 1, 1),  # 8, 256, 256
            nn.LeakyReLU(),
            nn.BatchNorm2d(8),
        )
        self.maxpool0 = nn.MaxPool2d(2, return_indices=True)  # 8, 128, 128
        self.deconv0 = nn.Sequential(
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 8, 3, 1, 1),  # 8, 256, 256
            nn.LeakyReLU(),
            nn.Conv2d(8, n_classes, 3, 1, 1),  # 1, 256, 256
        )
        self.maxunpool0 = nn.MaxUnpool2d(2)

        # Level 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(8, 16, 3, 1, 1),  # 16, 128, 128
            nn.LeakyReLU(),
            nn.BatchNorm2d(16),
        )
        self.deconv1 = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 8, 3, 1, 1),  # 16, 128, 128
            nn.LeakyReLU(),
            # nn.Conv2d(16, 16, 3, 1, 1),  # 16, 128, 128
            # nn.LeakyReLU(),
        )
        self.maxpool1 = nn.MaxPool2d(2, return_indices=True)
        self.maxunpool1 = nn.MaxUnpool2d(2)

        # Level 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 1),  # 32, 64, 64
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 16, 3, 1, 1),  # 64, 64, 64
            nn.LeakyReLU(),
        )

    def forward(self, x):
        # x: bs, nbands, 256, 256
        x0 = x
        x1 = self.conv0(x0)
        x1p5, ind0 = self.maxpool0(x1)
        x2 = self.conv1(x1p5)
        x2p5, ind1 = self.maxpool1(x2)
        x3 = self.conv2(x2p5)
        y1p5 = self.maxunpool1(x3, ind1)  # 128, 128
        y1 = self.deconv1(torch.cat([x2, y1p5], dim=1))
        y0p5 = self.maxunpool0(y1, ind0)
        y0 = self.deconv0(torch.cat([x1, y0p5], dim=1))
        if self.do_log_softmax:
            return F.log_softmax(y0, dim=1)
        else:
            return y0


if __name__ == '__main__':
    net = UNet128()
    net = net.cuda()
    x = torch.randn((2, 1, 128, 128), device='cuda')
    y = net(x)
    import pdb; pdb.set_trace()
