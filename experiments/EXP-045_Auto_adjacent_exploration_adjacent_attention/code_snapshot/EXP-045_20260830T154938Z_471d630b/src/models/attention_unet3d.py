"""
3D Wide Attention U-Net (AttentionUNet3D) with Spatial Attention Gates
Suppresses irrelevant background fluid while amplifying cellular nuclei features.
Reference: Oktay et al. - Attention U-Net: Learning Where to Look for the Pancreas
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

class AttentionGate3D(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # g: gating signal from deeper decoder layer
        # x: skip connection from encoder layer
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        # Spatial dimensions match after ConvTranspose
        psi = self.relu(g1 + x1)
        alpha = self.psi(psi)
        return x * alpha

class AttentionUNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=64):
        super().__init__()
        # Encoder
        self.enc1 = AnisotropicConvBlock(in_channels, base_channels, anisotropic=True)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.enc2 = AnisotropicConvBlock(base_channels, base_channels * 2, anisotropic=False)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.enc3 = AnisotropicConvBlock(base_channels * 2, base_channels * 4, anisotropic=False)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = AnisotropicConvBlock(base_channels * 4, base_channels * 8, anisotropic=False)

        # Attention Gates & Decoder
        self.up3 = nn.ConvTranspose3d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.att3 = AttentionGate3D(F_g=base_channels * 4, F_l=base_channels * 4, F_int=base_channels * 2)
        self.dec3 = AnisotropicConvBlock(base_channels * 8, base_channels * 4, anisotropic=False)

        self.up2 = nn.ConvTranspose3d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.att2 = AttentionGate3D(F_g=base_channels * 2, F_l=base_channels * 2, F_int=base_channels)
        self.dec2 = AnisotropicConvBlock(base_channels * 4, base_channels * 2, anisotropic=False)

        self.up1 = nn.ConvTranspose3d(base_channels * 2, base_channels, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.att1 = AttentionGate3D(F_g=base_channels, F_l=base_channels, F_int=base_channels // 2)
        self.dec1 = AnisotropicConvBlock(base_channels * 2, base_channels, anisotropic=True)

        self.head = nn.Conv3d(base_channels, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        b = self.bottleneck(p3)

        d3 = self.up3(b)
        e3_att = self.att3(g=d3, x=e3)
        d3 = self.dec3(torch.cat([d3, e3_att], dim=1))

        d2 = self.up2(d3)
        e2_att = self.att2(g=d2, x=e2)
        d2 = self.dec2(torch.cat([d2, e2_att], dim=1))

        d1 = self.up1(d2)
        e1_att = self.att1(g=d1, x=e1)
        d1 = self.dec1(torch.cat([d1, e1_att], dim=1))

        return torch.sigmoid(self.head(d1))
