"""
3D Residual U-Net (ResUNet3D) with Proven Feature Concatenation & Batch Normalization
Combines deep residual learning with robust spatial multi-scale skip connections.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        res = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + res)

class ResUNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=32):
        super().__init__()
        # Stem
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True)
        )
        self.enc1 = ResBlock3D(base_channels, base_channels)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.enc2 = ResBlock3D(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.enc3 = ResBlock3D(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = ResBlock3D(base_channels * 4, base_channels * 8)

        # Decoder with Proven Concatenation
        self.up3 = nn.ConvTranspose3d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv3d(base_channels * 8, base_channels * 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(base_channels * 4),
            nn.ReLU(inplace=True),
            ResBlock3D(base_channels * 4, base_channels * 4)
        )

        self.up2 = nn.ConvTranspose3d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv3d(base_channels * 4, base_channels * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(base_channels * 2),
            nn.ReLU(inplace=True),
            ResBlock3D(base_channels * 2, base_channels * 2)
        )

        self.up1 = nn.ConvTranspose3d(base_channels * 2, base_channels, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dec1 = nn.Sequential(
            nn.Conv3d(base_channels * 2, base_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True),
            ResBlock3D(base_channels, base_channels)
        )

        self.head = nn.Conv3d(base_channels, out_channels, kernel_size=1)

    def forward(self, x):
        s = self.stem(x)
        e1 = self.enc1(s)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        b = self.bottleneck(p3)

        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return torch.sigmoid(self.head(d1))
