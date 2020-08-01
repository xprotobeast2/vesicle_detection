import numpy as np
import torch
from torch import nn

class EncodeBlock(nn.Module):
    def __init__(self, in_channels=1, intermediate_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1):
        super(EncodeBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, intermediate_channels, kernel_size, stride, padding), 
            nn.LeakyReLU(),
            nn.Conv2d(intermediate_channels, out_channels, kernel_size, stride, padding),
            nn.LeakyReLU(),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.block(x)

class DecodeBlock(nn.Module):
    def __init__(self, in_channels=1, intermediate_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1):
        super(DecodeBlock, self).__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, intermediate_channels, kernel_size, stride, padding), 
            nn.LeakyReLU(),
            nn.Conv2d(intermediate_channels, out_channels, kernel_size, stride, padding),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.block(x)

class BottleneckBlock(nn.Module):
    def __init__(self, in_channels=1, intermediate_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1):
        super(BottleneckBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, intermediate_channels, kernel_size, stride, padding),
            nn.LeakyReLU(),
            nn.BatchNorm2d(intermediate_channels),
            nn.Conv2d(intermediate_channels, intermediate_channels, kernel_size, stride, padding),
            nn.LeakyReLU(),
            nn.Conv2d(intermediate_channels, out_channels, kernel_size, stride, padding),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.block(x)

class UNetAuto(nn.Module):
    """
    A parametrized UNet Model.

    """
    def __init__(self, in_channels=1, n_classes=2, depth=2, base_channels=16, 
        feat_scale=1, kernel_size=3, stride=1, padding=1):
        """
        Constructs the UNet with an encoding path, bottleneck, and decoding path.
        we use Conv2D, MaxPool and MaxUnpool. (Perhaps transposed convolutions can be 
        experimented with). Consult EncodeBlock, DecodeBlock, Bottleneck 
    
        params:

            in_channels: the number of channels in the input image
            n_classes: number of segmentation classes, also the number of channels in output layer
            depth: the number of blocks in the encoding or decoding path
            base_channels: The base number of feature maps after the input
            feat_scale: The multiplier for the number of feature maps in each consecutive block
            kernel_size: the width of the convolution kernel
            stride: convolution stride
            padding: convolution padding
        
        """



        super(UNetAuto, self).__init__()
        
        self.depth = depth

        # Collect layers
        self.convs = nn.ModuleList()
        self.deconvs = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.unpools = nn.ModuleList()

        self.convs.append(EncodeBlock(in_channels, base_channels, base_channels, kernel_size, stride, padding))

        # The factor of two is because of the concatenation between corresponding levels in the encoding and decoding paths
        self.deconvs.append(
            nn.Sequential(
                nn.BatchNorm2d(2*base_channels),
                nn.Conv2d(2*base_channels, base_channels, kernel_size, stride, padding), 
                nn.LeakyReLU(),
                nn.Conv2d(base_channels, n_classes, kernel_size, stride, padding)
            )
        )

        # Creat the max pool layers
        for i in range(depth):
            self.pools.append(nn.MaxPool2d(2, return_indices=True))
            self.unpools.append(nn.MaxUnpool2d(2))

        # Encoding and Decoding paths
        for i in range(1,depth):
            self.convs.append(EncodeBlock(base_channels*(feat_scale**(i-1)),base_channels*(feat_scale**i),base_channels*(feat_scale**i), kernel_size, stride, padding))
            self.deconvs.append(DecodeBlock(2*base_channels*(feat_scale**i), base_channels*(feat_scale**(i-1)), base_channels*(feat_scale**(i-1)), kernel_size, stride, padding))

        # Bottleneck path
        self.bottleneck = BottleneckBlock(base_channels*(feat_scale**(depth-1)), 2*base_channels*(feat_scale**(depth-1)), base_channels*(feat_scale**(depth-1)), kernel_size, stride, padding)

        # Softmax
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        xip = x
        xs = []
        indices = []
        # Encode
        for i in range(self.depth):
            xi = self.convs[i](xip)
            xs.append(xi)
            xip, inds = self.pools[i](xi)
            indices.append(inds)

        # Bottleneck
        yi = self.bottleneck(xip)


        # Decode
        for i in reversed(range(self.depth)):
            yip = self.unpools[i](yi, indices[i])
            yi = self.deconvs[i](torch.cat([xs[i], yip], dim=1))
        
        return self.softmax(yi)

class ConvBlock(nn.Module):
    def __init__(self, in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.LeakyReLU(),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.block(x)

class ResBlock(nn.Module):
    def __init__(self, in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        )

    def forward(self, x):
        fx = self.block(x)
        return x + fx

class FullResBlock(nn.Module):
    def __init__(self, in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1):
        super(FullResBlock, self).__init__()
        self.block = nn.Sequential(
            ConvBlock(in_channels, out_channels, kernel_size, stride, padding),
            ResBlock(out_channels, out_channels, kernel_size, stride, padding),
            ConvBlock(out_channels, out_channels, kernel_size, stride, padding)
        )

    def forward(self, x):
        return self.block(x)

class FRUNetAuto(nn.Module):
    """
    A parametrized Fully Residual UNet Model based on FusionNet 
    https://arxiv.org/pdf/1612.05360.pdf
    
    """


    def __init__(self, in_channels=1, n_classes=2, depth=2, base_channels=32, 
        feat_scale=1, kernel_size=3, stride=1, padding=1):
        """
        Constructs a UNet composed of residual blocks and skip connections 
        with an encoding path, bottleneck, and decoding path.
        we use Conv2D, MaxPool and MaxUnpool. (Perhaps transposed convolutions can be 
        experimented with). 

        params:

            in_channels: the number of channels in the input image
            n_classes: number of segmentation classes, also the number of channels in output layer
            depth: the number of blocks in the encoding or decoding path
            base_channels: The base number of feature maps after the input
            feat_scale: The multiplier for the number of feature maps in each consecutive block
            kernel_size: the width of the convolution kernel
            stride: convolution stride
            padding: convolution padding

        """

        super(FRUNetAuto, self).__init__()
        
        self.depth = depth

        # Collect layers
        self.convs = nn.ModuleList()
        self.deconvs = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.unpools = nn.ModuleList()

        self.convs.append(FullResBlock(in_channels, base_channels, kernel_size, stride, padding))

        # The factor of two is because of the concatenation between corresponding levels in the encoding and decoding paths
        self.deconvs.append(FullResBlock(base_channels, n_classes, kernel_size, stride, padding))

        # Creat the max pool layers
        for i in range(depth):
            self.pools.append(nn.MaxPool2d(2, return_indices=True))
            self.unpools.append(nn.MaxUnpool2d(2))

        # Encoding and Decoding paths
        for i in range(1, depth):
            self.convs.append(FullResBlock(base_channels*(feat_scale**(i-1)),base_channels*(feat_scale**i), kernel_size, stride, padding))
            self.deconvs.append(FullResBlock(base_channels*(feat_scale**i), base_channels*(feat_scale**(i-1)), kernel_size, stride, padding))

        # Bottleneck path
        self.bottleneck = FullResBlock(base_channels*(feat_scale**(depth-1)), base_channels*(feat_scale**(depth-1)), kernel_size, stride, padding)

        # Softmax
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        xip = x
        xs = []
        indices = []
        # Encode
        for i in range(self.depth):
            xi = self.convs[i](xip)
            xs.append(xi)
            xip, inds = self.pools[i](xi)
            indices.append(inds)

        # Bottleneck
        yi = self.bottleneck(xip)


        # Decode
        for i in reversed(range(self.depth)):
            yip = self.unpools[i](yi, indices[i])
            yi = self.deconvs[i](xs[i] + yip)
        
        return self.softmax(yi)


if __name__ == '__main__':
    net = UNet128()
    net = net.cuda()
    x = torch.randn((2, 1, 128, 128), device='cuda')
    y = net(x)
    import pdb; pdb.set_trace()

