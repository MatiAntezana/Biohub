"""
StarDist 3D Polyhedra Architecture for Cell Segmentation & Center Regression
Predicts object probability + 32 star-convex radial distances in 3D.
Reference: Weigert et al. (WACV) - Star-convex Polyhedra for 3D Cell Detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Generate 32 uniformly distributed vectors on the unit sphere (Fibonacci sphere)
def generate_sphere_rays(n_rays=32):
    indices = np.arange(0, n_rays, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / n_rays)
    theta = np.pi * (1 + 5**0.5) * indices
    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)
    rays = np.stack([z, y, x], axis=1).astype(np.float32) # (n_rays, 3) in (Z, Y, X)
    return rays

class StarDist3D(nn.Module):
    def __init__(self, in_channels=1, n_rays=32, base_channels=32):
        super().__init__()
        self.n_rays = n_rays
        rays = generate_sphere_rays(n_rays)
        self.register_buffer('rays', torch.from_numpy(rays))

        # 3D Anisotropic Encoder
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

        self.up2 = nn.ConvTranspose3d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv3d(base_channels * 4, base_channels * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(base_channels * 2),
            nn.ReLU(inplace=True)
        )

        self.up1 = nn.ConvTranspose3d(base_channels * 2, base_channels, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dec1 = nn.Sequential(
            nn.Conv3d(base_channels * 2, base_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True)
        )

        # Output Heads
        # Head 1: Cell probability (1 channel)
        self.head_prob = nn.Conv3d(base_channels, 1, kernel_size=1)
        # Head 2: Radial distances along 32 directions (in physical microns)
        self.head_dist = nn.Conv3d(base_channels, n_rays, kernel_size=1)

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

        prob = torch.sigmoid(self.head_prob(d1))
        dist = F.relu(self.head_dist(d1)) # Distances must be non-negative

        return {
            'prob': prob,
            'dist': dist
        }
