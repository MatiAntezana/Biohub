"""
3D ConvNeXt-UNet Architecture with Large 7x7 Depthwise Convolutions
Provides ultra-wide receptive field for deep Z-slices with strong optical dispersion.
Reference: Liu et al. (CVPR) - A ConvNet for the 2020s
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNeXtBlock3D(nn.Module):
    def __init__(self, dim, anisotropic=False):
        super().__init__()
        # Depthwise 3D convolution with large receptive field (3x7x7 or 7x7x7)
        k = (3, 7, 7) if anisotropic else (7, 7, 7)
        p = (1, 3, 3) if anisotropic else (3, 3, 3)
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=k, padding=p, groups=dim)
        self.norm = nn.BatchNorm3d(dim)
        # Pointwise 1x1 convolutions (Inverted Bottleneck 4x)
        self.pwconv1 = nn.Conv3d(dim, 4 * dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv3d(4 * dim, dim, kernel_size=1)
        self.gamma = nn.Parameter(1e-6 * torch.ones((1, dim, 1, 1, 1)), requires_grad=True)

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.gamma * x
        return input + x

class ConvNeXtUNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=32):
        super().__init__()
        # Stem
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(base_channels),
            nn.GELU()
        )
        self.enc1 = ConvNeXtBlock3D(base_channels, anisotropic=True)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.down2 = nn.Conv3d(base_channels, base_channels * 2, kernel_size=3, padding=1)
        self.enc2 = ConvNeXtBlock3D(base_channels * 2, anisotropic=False)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.down3 = nn.Conv3d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1)
        self.enc3 = ConvNeXtBlock3D(base_channels * 4, anisotropic=False)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        # Bottleneck
        self.down_b = nn.Conv3d(base_channels * 4, base_channels * 8, kernel_size=3, padding=1)
        self.bottleneck = ConvNeXtBlock3D(base_channels * 8, anisotropic=False)

        # Decoder with Proven Concatenation
        self.up3 = nn.ConvTranspose3d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv3d(base_channels * 8, base_channels * 4, kernel_size=3, padding=1),
            ConvNeXtBlock3D(base_channels * 4, anisotropic=False)
        )

        self.up2 = nn.ConvTranspose3d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv3d(base_channels * 4, base_channels * 2, kernel_size=3, padding=1),
            ConvNeXtBlock3D(base_channels * 2, anisotropic=False)
        )

        self.up1 = nn.ConvTranspose3d(base_channels * 2, base_channels, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dec1 = nn.Sequential(
            nn.Conv3d(base_channels * 2, base_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            ConvNeXtBlock3D(base_channels, anisotropic=True)
        )

        self.head = nn.Conv3d(base_channels, out_channels, kernel_size=1)

    def forward(self, x):
        s = self.stem(x)
        e1 = self.enc1(s)
        p1 = self.pool1(e1)

        e2 = self.enc2(self.down2(p1))
        p2 = self.pool2(e2)

        e3 = self.enc3(self.down3(p2))
        p3 = self.pool3(e3)

        b = self.bottleneck(self.down_b(p3))

        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return torch.sigmoid(self.head(d1))
