"""
3D Full-Volume GroupNorm U-Net (GroupNormUNet3D)
Replaces BatchNorm3d with GroupNorm3d (num_groups=8) to ensure 100% numerical stability
when training on massive full embryo volumes (64, 256, 256) with batch_size=1 or batch_size=2.
Zero running_mean/running_var tracking, eliminating NaN risk completely.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

class GroupNormConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, anisotropic=True, num_groups=8):
        super().__init__()
        k1 = (1, 3, 3) if anisotropic else (3, 3, 3)
        p1 = (0, 1, 1) if anisotropic else (1, 1, 1)
        
        # Ensure num_groups divides channels
        groups_in = min(num_groups, out_channels)
        while out_channels % groups_in != 0:
            groups_in -= 1

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=k1, padding=p1, bias=False)
        self.gn1 = nn.GroupNorm(num_groups=groups_in, num_channels=out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(num_groups=groups_in, num_channels=out_channels)

    def forward(self, x):
        x = self.relu(self.gn1(self.conv1(x)))
        x = self.relu(self.gn2(self.conv2(x)))
        return x

class GroupNormUNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=64, gradient_checkpointing=True):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing

        # Encoder
        self.enc1 = GroupNormConvBlock(in_channels, base_channels, anisotropic=True)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.enc2 = GroupNormConvBlock(base_channels, base_channels * 2, anisotropic=False)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.enc3 = GroupNormConvBlock(base_channels * 2, base_channels * 4, anisotropic=False)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = GroupNormConvBlock(base_channels * 4, base_channels * 8, anisotropic=False)

        # Decoder
        self.up3 = nn.ConvTranspose3d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = GroupNormConvBlock(base_channels * 8, base_channels * 4, anisotropic=False)

        self.up2 = nn.ConvTranspose3d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = GroupNormConvBlock(base_channels * 4, base_channels * 2, anisotropic=False)

        self.up1 = nn.ConvTranspose3d(base_channels * 2, base_channels, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dec1 = GroupNormConvBlock(base_channels * 2, base_channels, anisotropic=True)

        self.head = nn.Conv3d(base_channels, out_channels, kernel_size=1)

    def forward(self, x):
        if self.gradient_checkpointing and self.training:
            e1 = checkpoint(self.enc1, x, use_reentrant=False)
            p1 = self.pool1(e1)
            e2 = checkpoint(self.enc2, p1, use_reentrant=False)
            p2 = self.pool2(e2)
            e3 = checkpoint(self.enc3, p2, use_reentrant=False)
            p3 = self.pool3(e3)
            b = checkpoint(self.bottleneck, p3, use_reentrant=False)
        else:
            e1 = self.enc1(x)
            p1 = self.pool1(e1)
            e2 = self.enc2(p1)
            p2 = self.pool2(e2)
            e3 = self.enc3(p2)
            p3 = self.pool3(e3)
            b = self.bottleneck(p3)

        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        out = self.head(d1)
        return torch.sigmoid(out)
