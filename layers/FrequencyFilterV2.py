"""
FrequencyFilterV2 — Complex-Weight Spectral Filter

Difference from V1 (FrequencyFilter.py):
  V1: sigmoid(W_real) ⊙ FFT(x)  →  magnitude scaling only, phase preserved
  V2: (sigmoid(W_real) * e^{j·tanh(W_imag)·π}) ⊙ FFT(x)  →  magnitude + phase

Why complex weights:
  RevIN normalizes each window independently ((x - mean) / std), which
  compresses magnitude differences across frequencies. But RevIN does NOT
  destroy the phase relationships between frequency components (e.g., the
  alignment between the daily cycle and its 12h harmonic).

  By adding learnable phase shifts, the module can:
  - Correct misalignment between harmonics introduced by noise
  - Adjust the relative timing of sub-harmonic components
  - Provide a richer parameterization than magnitude-only filtering

Key properties:
  - O(2 × C × N_freq) parameters (twice V1, still O(C))
  - Zero-init: starts as identity via res_scale=0
  - No LayerNorm/Dropout: preserves identity property
  - V1 kept for comparison; controlled via --freq_v2_layers

Layer count controlled via --freq_v2_layers (default=0 = disabled).
"""

import torch
import torch.nn as nn
import numpy as np


class FrequencyFilterV2Layer(nn.Module):
    """
    Per-channel learnable complex frequency mask.

    Input:  [B, C, L]
    Output: [B, C, L]  (zero-init residual: x + res_scale * filtered)

    Architecture:
        x → rfft → X_complex [B, C, N_freq]
               → W = sigmoid(W_real) * exp(j * tanh(W_imag) * pi)
               → X_filtered = X * W
               → irfft → filtered
               → x + res_scale × filtered   ← zero-init identity

    Parameters: 2 × C × N_freq + 1
      - ETTh1 (C=7):  2×7×49+1 = 687 params
      - Traffic (C=862): 2×862×49+1 = 84,477 params
    """

    def __init__(self, seq_len, channel_size):
        super().__init__()
        self.seq_len = seq_len
        self.n_freq = seq_len // 2 + 1       # rfft bins (e.g., 49 for L=96)
        self.channel_size = channel_size

        # Magnitude weights: sigmoid(0) = 0.5 → initially neutral
        self.freq_weights_real = nn.Parameter(torch.zeros(channel_size, self.n_freq))

        # Phase weights: tanh(0) = 0 → initially no phase shift
        self.freq_weights_imag = nn.Parameter(torch.zeros(channel_size, self.n_freq))

        # Zero-init: module starts as identity, gradually contributes
        self.res_scale = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # x: [B, C, L]
        B, C, L = x.shape

        # Real FFT along time dimension
        X = torch.fft.rfft(x, dim=-1)           # [B, C, N_freq] complex

        # Complex weight: magnitude ∈ (0,1), phase ∈ [-π, π]
        mag = torch.sigmoid(self.freq_weights_real)           # [C, N_freq]
        phase = torch.tanh(self.freq_weights_imag) * np.pi    # [C, N_freq]
        W = mag * torch.exp(1j * phase)                       # [C, N_freq] complex

        # Apply complex weight
        X_filtered = X * W.unsqueeze(0)          # broadcast over batch

        # Inverse FFT back to time domain
        out = torch.fft.irfft(X_filtered, n=L, dim=-1)  # [B, C, L]

        # Zero-init residual
        return x + self.res_scale * out
