"""
3D Anisotropic U-Net with Multi-Scale Deep Supervision
Computes auxiliary loss heads at Dec3, Dec2, Dec1 to enforce sharp hierarchical gradient backpropagation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepSupervisionUNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=32):
        super().__init__()
        # Encoder
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Sequential(
            nn.Conv3d(base_channels, base_channels * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(base_channels * 2),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv3 = nn.Sequential(
            nn.Conv3d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_channels * 4, base_channels * 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(base_channels * 4),
            nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv3d(base_channels * 4, base_channels * 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(base_channels * 8),
            nn.ReLU(inplace=True)
        )

        # Decoder
        self.up3 = nn.ConvTranspose3d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv3d(base_channels * 8, base_channels * 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(base_channels * 4),
            nn.ReLU(inplace=True)
        )
        self.head_aux3 = nn.Conv3d(base_channels * 4, out_channels, kernel_size=1)

        self.up2 = nn.ConvTranspose3d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv3d(base_channels * 4, base_channels * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(base_channels * 2),
            nn.ReLU(inplace=True)
        )
        self.head_aux2 = nn.Conv3d(base_channels * 2, out_channels, kernel_size=1)

        self.up1 = nn.ConvTranspose3d(base_channels * 2, base_channels, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dec1 = nn.Sequential(
            nn.Conv3d(base_channels * 2, base_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True)
        )
        self.head_main = nn.Conv3d(base_channels, out_channels, kernel_size=1)

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)

        c2 = self.conv2(p1)
        p2 = self.pool2(c2)

        c3 = self.conv3(p2)
        p3 = self.pool3(c3)

        b = self.bottleneck(p3)

        d3 = self.dec3(torch.cat([self.up3(b), c3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), c2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), c1], dim=1))

        main_out = torch.sigmoid(self.head_main(d1))

        if self.training:
            # Auxiliary supervision heads upsampled to input size
            aux2 = torch.sigmoid(F.interpolate(self.head_aux2(d2), size=x.shape[2:], mode='trilinear', align_corners=False))
            aux3 = torch.sigmoid(F.interpolate(self.head_aux3(d3), size=x.shape[2:], mode='trilinear', align_corners=False))
            return {
                'main': main_out,
                'aux2': aux2,
                'aux3': aux3
            }
        return main_out
