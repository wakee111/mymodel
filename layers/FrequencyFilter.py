"""
FrequencyFilter — Frequency-Domain Parameterized Spectral Filter

Motivation:
  After RecurrentCycle removes the base frequency (f₀), the residual still
  contains significant energy at harmonic frequencies (2f₀, 3f₀, 4f₀, ...).
  These sub-harmonic components are invisible to:
  - RecurrentCycle (single base frequency, time-domain lookup)
  - MRT (avg_pool is a low-pass filter, suppresses mid/high frequencies)
  - MLP (global but no frequency-structure inductive bias)

  This module provides a learnable per-channel frequency mask:
  x → FFT → W ⊙ FFT(x) → IFFT → filtered output

  FFT/IFFT is lossless — no information is discarded, only re-weighted.
  This is fundamentally different from MRT's avg_pool which discards
  high-frequency detail during downsampling.

Key properties:
  - Equivalent to a global convolution in time domain
  - O(C × N_freq) parameters (N_freq = L//2+1 = 49 for L=96)
  - Zero-init: starts as identity via res_scale=0
  - No LayerNorm/Dropout: preserves identity property
  - Complements MRT: targets different frequency bands

Inspired by:
  - FEDformer (ICML 2022): frequency-enhanced attention, using FFT for
    global receptive field in Transformers
  - The core difference: we apply frequency filtering on the residual
    (not the original signal), targeting sub-harmonic components that
    RecurrentCycle cannot model

Layer count controlled via --freq_layers (default=0 = disabled).
"""

import torch
import torch.nn as nn


class FrequencyFilterLayer(nn.Module):
    """
    Per-channel learnable frequency mask.

    Input:  [B, C, L]
    Output: [B, C, L]  (zero-init residual: x + res_scale * filtered)

    Architecture:
        x → rfft → [B, C, N_freq]
               → × sigmoid(W[C, N_freq])   ← learnable per-channel mask
               → irfft → filtered
               → x + res_scale × filtered   ← zero-init identity

    Parameters: C × N_freq + 1
      - ETTh1 (C=7):  7×49+1 = 344 params
      - Traffic (C=862): 862×49+1 = 42,239 params
    """

    def __init__(self, seq_len, channel_size):
        super().__init__()
        self.seq_len = seq_len
        self.n_freq = seq_len // 2 + 1       # rfft bins (e.g., 49 for L=96)
        self.channel_size = channel_size

        # Per-channel frequency weights, initialized to 0.
        # sigmoid(0) = 0.5 → initially neutral (neither enhance nor suppress).
        # During training, important frequencies are pushed toward 1,
        # unimportant toward 0.
        self.freq_weights = nn.Parameter(torch.zeros(channel_size, self.n_freq))

        # Zero-init: module starts as identity, gradually contributes
        self.res_scale = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # x: [B, C, L]
        B, C, L = x.shape

        # Real FFT along time dimension
        X = torch.fft.rfft(x, dim=-1)           # [B, C, N_freq] complex64

        # Per-channel frequency mask [C, N_freq] → [1, C, N_freq]
        mask = torch.sigmoid(self.freq_weights)  # (0, 1)

        # Apply mask to magnitude (multiply complex by real scalar)
        X_filtered = X * mask.unsqueeze(0)       # broadcast over batch

        # Inverse FFT back to time domain
        out = torch.fft.irfft(X_filtered, n=L, dim=-1)  # [B, C, L]

        # Zero-init residual: x + 0 * out = x at initialization
        return x + self.res_scale * out
