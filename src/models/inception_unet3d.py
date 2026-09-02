"""
3D Inception-UNet with Multi-Scale Dilated Kernels (Dilation 1, 2, 3)
Captures both tight sub-cellular centroid peaks and wide blastomere context.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class InceptionBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        branch_ch = out_channels // 4
        
        # Branch 1: 1x1x1 conv
        self.b1 = nn.Sequential(
            nn.Conv3d(in_channels, branch_ch, kernel_size=1, bias=False),
            nn.BatchNorm3d(branch_ch),
            nn.ReLU(inplace=True)
        )
        
        # Branch 2: 1x3x3 conv (anisotropic)
        self.b2 = nn.Sequential(
            nn.Conv3d(in_channels, branch_ch, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(branch_ch),
            nn.ReLU(inplace=True)
        )
        
        # Branch 3: 3x3x3 conv with dilation=2
        self.b3 = nn.Sequential(
            nn.Conv3d(in_channels, branch_ch, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm3d(branch_ch),
            nn.ReLU(inplace=True)
        )
        
        # Branch 4: 3x3x3 standard conv
        self.b4 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels - 3 * branch_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels - 3 * branch_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)

class InceptionUNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=32):
        super().__init__()
        self.enc1 = InceptionBlock3D(in_channels, base_channels)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        
        self.enc2 = InceptionBlock3D(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.enc3 = InceptionBlock3D(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.bottleneck = InceptionBlock3D(base_channels * 4, base_channels * 8)
        
        self.up3 = nn.ConvTranspose3d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = InceptionBlock3D(base_channels * 8, base_channels * 4)
        
        self.up2 = nn.ConvTranspose3d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = InceptionBlock3D(base_channels * 4, base_channels * 2)
        
        self.up1 = nn.ConvTranspose3d(base_channels * 2, base_channels, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dec1 = InceptionBlock3D(base_channels * 2, base_channels)
        
        self.head = nn.Conv3d(base_channels, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
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
