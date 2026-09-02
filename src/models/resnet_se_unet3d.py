"""
3D Residual U-Net with Squeeze-and-Excitation (SE-Block) Channel Attention
Numerically stable, O(C) complexity, and highly effective for 3D microscopy.
Reference: Hu et al. (CVPR) - Squeeze-and-Excitation Networks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock3D(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Conv3d(channels, channels // reduction, kernel_size=1)
        self.fc2 = nn.Conv3d(channels // reduction, channels, kernel_size=1)

    def forward(self, x):
        # Global Average Pooling in 3D: (B, C, Z, Y, X) -> (B, C, 1, 1, 1)
        w = torch.mean(x, dim=(-3, -2, -1), keepdim=True)
        w = F.relu(self.fc1(w), inplace=True)
        w = torch.sigmoid(self.fc2(w))
        return x * w

class SEResBlock3D(nn.Module):
    def __init__(self, channels, anisotropic=False):
        super().__init__()
        k = (1, 3, 3) if anisotropic else (3, 3, 3)
        p = (0, 1, 1) if anisotropic else (1, 1, 1)
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=k, padding=p, bias=False)
        self.bn1 = nn.BatchNorm3d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(channels)
        self.se = SEBlock3D(channels, reduction=8)

    def forward(self, x):
        res = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        return self.relu(out + res)

class SEResNetUNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=32):
        super().__init__()
        # Stem
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True)
        )
        self.enc1 = SEResBlock3D(base_channels, anisotropic=True)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.down2 = nn.Conv3d(base_channels, base_channels * 2, kernel_size=3, padding=1)
        self.enc2 = SEResBlock3D(base_channels * 2, anisotropic=False)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.down3 = nn.Conv3d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1)
        self.enc3 = SEResBlock3D(base_channels * 4, anisotropic=False)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        # Bottleneck
        self.down_b = nn.Conv3d(base_channels * 4, base_channels * 8, kernel_size=3, padding=1)
        self.bottleneck = SEResBlock3D(base_channels * 8, anisotropic=False)

        # Decoder
        self.up3 = nn.ConvTranspose3d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = SEResBlock3D(base_channels * 4, anisotropic=False)

        self.up2 = nn.ConvTranspose3d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = SEResBlock3D(base_channels * 2, anisotropic=False)

        self.up1 = nn.ConvTranspose3d(base_channels * 2, base_channels, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dec1 = SEResBlock3D(base_channels, anisotropic=True)

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

        d3 = self.dec3(self.up3(b) + e3)
        d2 = self.dec2(self.up2(d3) + e2)
        d1 = self.dec1(self.up1(d2) + e1)

        return torch.sigmoid(self.head(d1))
