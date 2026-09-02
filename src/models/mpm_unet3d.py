"""
Motion and Position Map (MPM) 3D Anisotropic U-Net
Predicts cell centers, 3D forward/backward displacement vectors, and division probability.
Reference: Hayashida et al. (CVPR 2020 & MICCAI)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class AnisotropicConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, anisotropic=True):
        super().__init__()
        k1 = (1, 3, 3) if anisotropic else (3, 3, 3)
        p1 = (0, 1, 1) if anisotropic else (1, 1, 1)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=k1, padding=p1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

class MPMUNet3D(nn.Module):
    def __init__(self, in_channels=1, base_channels=32):
        super().__init__()
        # Shared Encoder
        self.enc1 = AnisotropicConvBlock(in_channels, base_channels, anisotropic=True)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.enc2 = AnisotropicConvBlock(base_channels, base_channels * 2, anisotropic=False)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.enc3 = AnisotropicConvBlock(base_channels * 2, base_channels * 4, anisotropic=False)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.bottleneck = AnisotropicConvBlock(base_channels * 4, base_channels * 8, anisotropic=False)

        # Shared Decoder
        self.up3 = nn.ConvTranspose3d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = AnisotropicConvBlock(base_channels * 8, base_channels * 4, anisotropic=False)

        self.up2 = nn.ConvTranspose3d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = AnisotropicConvBlock(base_channels * 4, base_channels * 2, anisotropic=False)

        self.up1 = nn.ConvTranspose3d(base_channels * 2, base_channels, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dec1 = AnisotropicConvBlock(base_channels * 2, base_channels, anisotropic=True)

        # Multi-Task Output Heads
        # Head 1: Heatmap of cell centers (1 channel, sigmoid)
        self.head_heat = nn.Conv3d(base_channels, 1, kernel_size=1)

        # Head 2: Forward 3D Flow Vector (vz, vy, vx) (3 channels, linear)
        self.head_flow_fwd = nn.Conv3d(base_channels, 3, kernel_size=1)

        # Head 3: Mitosis probability map (1 channel, sigmoid)
        self.head_div = nn.Conv3d(base_channels, 1, kernel_size=1)

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
        feat = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        heat = torch.sigmoid(self.head_heat(feat))
        flow_fwd = self.head_flow_fwd(feat) # In physical microns (dz, dy, dx)
        div = torch.sigmoid(self.head_div(feat))

        return {
            'heatmap': heat,
            'flow_fwd': flow_fwd,
            'mitosis': div,
            'features': feat
        }
