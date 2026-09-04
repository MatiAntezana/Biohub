"""
PyTorch model architecture definitions for Biohub Autoresearch V2.
Directly compatible with EXP-056 champion checkpoint weights.
"""

from __future__ import annotations
import math
from collections.abc import Sequence
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block_3d(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True),
    )


class TemporalAttention(nn.Module):
    """Per-voxel multi-head self-attention across time."""

    def __init__(self, channels: int, n_heads: int = 4) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, n_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape[:3]
        spatial = x.shape[3:]
        S = math.prod(spatial)

        h = x.reshape(B, T, C, S).permute(0, 3, 1, 2).reshape(B * S, T, C)
        h = self.norm(h)
        h, _ = self.attn(h, h, h, need_weights=False)
        h = h.reshape(B, S, T, C).permute(0, 2, 3, 1).reshape(B, T, C, *spatial)
        return x + h


class TemporalUNet3D(nn.Module):
    """Minimal 3D temporal U-Net."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 48,
        layers: Sequence[int] = (48, 96, 192),
        skip_fullres_temporal: bool = True,
    ) -> None:
        super().__init__()
        layers = list(layers)
        self.encoder_blocks = nn.ModuleList()
        self.temporal_blocks = nn.ModuleList()
        prev = in_channels
        for i, ch in enumerate(layers):
            self.encoder_blocks.append(conv_block_3d(prev, ch))
            if skip_fullres_temporal and i == 0:
                self.temporal_blocks.append(nn.Identity())
            else:
                self.temporal_blocks.append(TemporalAttention(ch))
            prev = ch
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        self.upsamples = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        for i in range(len(layers) - 1, 0, -1):
            self.upsamples.append(
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
            )
            self.decoder_blocks.append(conv_block_3d(layers[i] + layers[i - 1], layers[i - 1]))

        self.head = nn.Conv3d(layers[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape[:2]
        x_flat = x.reshape(B * T, *x.shape[2:])

        skips: list[torch.Tensor] = []
        for i, (block, temporal) in enumerate(zip(self.encoder_blocks, self.temporal_blocks)):
            if i > 0:
                x_flat = self.pool(x_flat)
            x_flat = block(x_flat)
            x_flat = temporal(x_flat.reshape(B, T, *x_flat.shape[1:])).reshape(B * T, *x_flat.shape[1:])
            if i < len(self.encoder_blocks) - 1:
                skips.append(x_flat)

        for up, block, skip in zip(self.upsamples, self.decoder_blocks, skips[::-1]):
            x_flat = up(x_flat)
            if x_flat.shape[2:] != skip.shape[2:]:
                x_flat = F.interpolate(x_flat, size=skip.shape[2:], mode="trilinear", align_corners=False)
            x_flat = torch.cat([x_flat, skip], dim=1)
            x_flat = block(x_flat)

        x_flat = self.head(x_flat)
        return x_flat.reshape(B, T, *x_flat.shape[1:])


class CrossAttentionBlock(nn.Module):
    """A single cross-attention block with MLP and residual connections."""

    def __init__(
        self,
        hidden_dim: int = 192,
        n_heads: int = 6,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, n_heads, batch_first=True, dropout=dropout
        )

        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        kv_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        key_padding_mask = ~kv_mask if kv_mask is not None else None
        attn_out, _ = self.cross_attn(
            self.norm1(q), self.norm1(kv), self.norm1(kv),
            key_padding_mask=key_padding_mask,
        )
        q = q + attn_out
        q = q + self.mlp(self.norm2(q))
        return q


class SimpleNodeTransformer(nn.Module):
    """Transformer for predicting edges between cell detections."""

    def __init__(
        self,
        feat_dim: int = 80,
        hidden_dim: int = 192,
        n_heads: int = 6,
        n_blocks: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
        pair_chunk_size: int | None = 32,
    ):
        super().__init__()
        self.pair_chunk_size = pair_chunk_size
        self.proj = nn.Linear(feat_dim, hidden_dim)
        self.norm_in = nn.LayerNorm(hidden_dim)

        self.blocks = nn.ModuleList([
            CrossAttentionBlock(hidden_dim, n_heads, mlp_ratio, dropout)
            for _ in range(n_blocks)
        ])

        self.norm_out = nn.LayerNorm(hidden_dim)

        self.pair_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        feat_src: torch.Tensor,
        feat_tgt: torch.Tensor,
        coords_src: torch.Tensor,
        coords_tgt: torch.Tensor,
        mask_src: torch.Tensor | None = None,
        mask_tgt: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h_src = self.norm_in(self.proj(feat_src))
        h_tgt = self.norm_in(self.proj(feat_tgt))

        for block in self.blocks:
            h_tgt = block(h_tgt, h_src, mask_src)

        h_src = self.norm_out(h_src)
        h_tgt = self.norm_out(h_tgt)

        B, N_src, D = h_src.shape
        _, N_tgt, _ = h_tgt.shape

        h_src_exp = h_src.unsqueeze(2).expand(B, N_src, N_tgt, D)
        h_tgt_exp = h_tgt.unsqueeze(1).expand(B, N_src, N_tgt, D)
        rel_pos = coords_tgt.unsqueeze(1) - coords_src.unsqueeze(2)

        pair_feats = torch.cat([h_src_exp, h_tgt_exp, rel_pos], dim=-1)
        logits = self.pair_mlp(pair_feats).squeeze(-1)
        return logits


class UNetNodeTransformer(nn.Module):
    """TemporalUNet3D encoder + detection head + SimpleNodeTransformer."""

    def __init__(
        self,
        unet_out_channels: int = 48,
        unet_layers: Sequence[int] = (48, 96, 192),
        hidden_dim: int = 192,
        n_heads: int = 6,
        n_blocks: int = 4,
        pos_feat_dim: int = 32,
    ):
        super().__init__()
        self.unet = TemporalUNet3D(
            in_channels=1,
            out_channels=unet_out_channels,
            layers=unet_layers,
        )
        self.detect_head = nn.Conv3d(unet_out_channels, 1, kernel_size=1)
        self.transformer = SimpleNodeTransformer(
            feat_dim=unet_out_channels + pos_feat_dim,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            n_blocks=n_blocks,
        )

    def detect(self, frame: torch.Tensor) -> torch.Tensor:
        if frame.dim() == 3:
            pair = torch.stack([frame, frame], dim=0).unsqueeze(0).unsqueeze(2)
        elif frame.dim() == 4:
            pair = torch.stack([frame[0], frame[0]], dim=0).unsqueeze(0).unsqueeze(2)
        else:
            pair = frame
        unet_out = self.unet(pair)
        det = self.detect_head(unet_out[0, 0:1])
        return det[0, 0]

    def encode(self, imgs: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        window = imgs.unsqueeze(2)
        unet_out = self.unet(window)
        W = unet_out.shape[1]
        det_logits = [self.detect_head(unet_out[:, i]) for i in range(W)]
        return unet_out, det_logits
