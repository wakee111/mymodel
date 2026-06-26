"""
MultiResidualTrend (MRT) — Multi-Resolution Residual Trend Block

Inspired by:
  - N-HiTS (2023): multi-rate signal sampling for hierarchical forecasting
  - N-BEATS (ICLR 2020): residual stacking with basis expansion
  - Wavelet decomposition in signal processing

Core insight:
  After cycle removal, the residual has low SNR at full resolution (96 steps),
  but trends become visible at coarser resolutions via downsampling.
  The MLP(96→512→pred) only sees the full-resolution residual — it misses the
  multi-scale trend structure that emerges when you look at the residual at
  different time granularities.

How it works:
  1. Downsample residual at multiple scales (avg_pool with stride)
  2. Extract trend features from each scale independently
  3. Upsample and fuse back to original length
  4. Zero-init residual: module starts as identity, gradually contributes

This module is specifically designed for the residual signal (not the original
time series), targeting the multi-scale trend structure that survives after
cycle removal.

Layer count controlled via --mrt_layers (default=0 = disabled).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiResidualTrendLayer(nn.Module):
    """
    Single MRT layer: multi-resolution analysis of residual signal.

    Three branches at different temporal resolutions:
      - Branch 1 (native): conv at full resolution, captures fine momentum
      - Branch 2 (mid):   avg_pool(2) → conv → upsample, captures mid-scale trend
      - Branch 3 (coarse): avg_pool(4) → conv → upsample, captures coarse trend

    Uses per-channel learnable fusion weights (O(C) params) instead of
    Conv1d projection (O(C²) params), preventing overfitting on high-channel datasets.
    """

    def __init__(self, seq_len, channel_size, scales=(1, 2, 4), kernel_size=3):
        super(MultiResidualTrendLayer, self).__init__()
        self.seq_len = seq_len
        self.channel_size = channel_size
        self.scales = scales

        # One depthwise conv per scale (preserves channel independence)
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=channel_size,
                out_channels=channel_size,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                groups=channel_size,  # depthwise: per-channel
                bias=False
            )
            for _ in scales
        ])

        # Per-channel learnable fusion weights (softmax across scales)
        # Shape: [C, len(scales)] → each channel learns which scale to trust
        # O(C) parameters instead of O(C²) from Conv1d projection
        self.channel_fusion = nn.Parameter(torch.zeros(channel_size, len(scales)))

        # Zero-init residual scale
        self.res_scale = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
        Args:
            x: [B, C, L]  (batch, channels, seq_len) — the residual signal
        Returns:
            [B, C, L] same shape
        """
        B, C, L = x.shape

        branch_outputs = []

        for i, (scale, conv) in enumerate(zip(self.scales, self.convs)):
            if scale == 1:
                branch_out = conv(x)  # [B, C, L]
            else:
                x_down = F.avg_pool1d(x, kernel_size=scale, stride=scale)
                out_down = conv(x_down)  # [B, C, L//scale]
                branch_out = F.interpolate(
                    out_down, size=L, mode='linear', align_corners=False
                )

            branch_outputs.append(branch_out)

        # Per-channel softmax fusion: [B, C, L, num_scales]
        stacked = torch.stack(branch_outputs, dim=-1)

        # Each channel learns its own scale preference: [C, num_scales]
        ch_w = torch.softmax(self.channel_fusion, dim=-1)

        # Weighted sum across scales: [B, C, L]
        x_out = (stacked * ch_w.unsqueeze(0).unsqueeze(2)).sum(dim=-1)

        # Zero-init residual: at init, res_scale=0 → output = x (identity)
        return x + self.res_scale * x_out
