"""
FrequencyFilterV4 — Frequency Band Basis (Decomposed Spectral Filter)

Difference from V1/V2/V3:
  V1: C × N_freq independent weights per channel per frequency
  V4: K basis filters (K << N_freq) × per-channel combination weights
      mask[C, N_freq] = softmax(alpha[C, K]) @ sigmoid(basis[K, N_freq])

Why band basis:
  Directly learning C × N_freq = 343 weights (for ETTh1, C=7) gives each
  frequency bin independent control. But real frequency structure is often
  band-like — nearby frequencies should be treated similarly.

  With K basis filters (e.g. K=4), each basis filter learns a frequency
  band pattern (low-pass, band-pass, high-pass, etc.), and each channel
  learns a combination of these filters. This introduces a structural
  inductive bias toward band-wise processing.

  Advantages:
  - More structured: bands instead of independent bins
  - Very compact: K × N_freq + C × K params (e.g., 4×49 + 7×4 = 224)
  - Interpretable: can visualize what each basis filter learned

Key properties:
  - O(K × N_freq + C × K) parameters (e.g., 224 for ETTh1 with K=4)
  - Zero-init: starts as identity via res_scale=0
  - Basis filters shared across channels
  - Per-channel mixing weights for personalized band preferences

Layer count controlled via --freq_v4_layers (default=0 = disabled).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FrequencyFilterV4Layer(nn.Module):
    """
    Band-basis frequency filter: K basis filters × per-channel mixing.

    Input:  [B, C, L]
    Output: [B, C, L]  (zero-init residual: x + res_scale * filtered)

    Architecture:
        x → rfft → X [B, C, N_freq]
               → basis[K, N_freq] = sigmoid(W_basis)      ← K learnable filters
               → alpha[C, K] = softmax(W_mix)             ← per-channel mixing
               → mask[C, N_freq] = alpha @ basis
               → X_filtered = X × mask
               → irfft → filtered → x + res_scale × filtered

    Parameters:
      - Basis: K × N_freq
      - Mixing: C × K
      - Total: K(N_freq + C)
    """

    def __init__(self, seq_len, channel_size, n_bands=4):
        super().__init__()
        self.seq_len = seq_len
        self.n_freq = seq_len // 2 + 1
        self.channel_size = channel_size
        self.n_bands = n_bands

        # K learnable basis filters [K, N_freq]
        # sigmoid(0)=0.5 initially (neutral, no frequency preference)
        self.basis = nn.Parameter(torch.zeros(n_bands, self.n_freq))

        # Per-channel combination weights [C, K]
        self.channel_mix = nn.Parameter(torch.zeros(channel_size, n_bands))

        # Zero-init
        self.res_scale = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, L = x.shape

        # FFT
        X = torch.fft.rfft(x, dim=-1)              # [B, C, N_freq] complex

        # Basis filters: each is a frequency pattern [K, N_freq]
        basis_w = torch.sigmoid(self.basis)          # [K, N_freq]

        # Per-channel mixing: softmax ensures sum=1 per channel
        alpha = F.softmax(self.channel_mix, dim=-1)  # [C, K]

        # Combined mask: weighted sum of basis filters
        mask = alpha @ basis_w                       # [C, N_freq]

        # Apply
        X_filtered = X * mask.unsqueeze(0)           # broadcast batch

        # IFFT
        out = torch.fft.irfft(X_filtered, n=L, dim=-1)  # [B, C, L]

        return x + self.res_scale * out
