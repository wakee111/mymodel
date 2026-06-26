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

    def __init__(self, seq_len, channel_size, res_scale_init=0.0,
                 low_freq_bias=-2.0, n_low_bins=8):
        """
        Args:
            low_freq_bias: logit bias for low-frequency bins.
                sigmoid(-2.0) ≈ 0.12 → Freq starts "blind" to low freqs,
                forcing it to find gradient signal in mid/high frequencies.
                MRT's avg_pool naturally handles the low end.
            n_low_bins: number of lowest frequency bins to bias.
                For seq_len=96 (n_freq=49), bins 0-7 cover periods > 12 samples.
        """
        super().__init__()
        self.seq_len = seq_len
        self.n_freq = seq_len // 2 + 1       # rfft bins (e.g., 49 for L=96)
        self.channel_size = channel_size
        self.low_freq_bias = low_freq_bias
        self.n_low_bins = min(n_low_bins, self.n_freq)

        # Per-channel frequency weights.
        # Low-freq bins: init with negative bias → sigmoid(-2.0) ≈ 0.12 (suppressed)
        # Mid/high-freq bins: init at 0 → sigmoid(0) = 0.50 (neutral, learnable)
        # This creates division of labor: MRT covers LF, Freq is pushed toward MF/HF.
        init_weights = torch.zeros(channel_size, self.n_freq)
        if self.n_low_bins > 0:
            init_weights[:, :self.n_low_bins] = low_freq_bias
        self.freq_weights = nn.Parameter(init_weights)

        # res_scale: init=0 for identity start, init<0 for "subtractive filter" start
        self.res_scale = nn.Parameter(torch.tensor(res_scale_init, dtype=torch.float32))

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
