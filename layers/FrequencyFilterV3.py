"""
FrequencyFilterV3 — Input-Adaptive (Dynamic) Spectral Mask

Difference from V1/V2:
  V1: static sigmoid(W) ⊙ FFT(x)  →  same mask for all windows
  V3: static_mask * adapt(mag(x)) ⊙ FFT(x)  →  mask adapts to each window

Why adaptive:
  Different input windows have different instantaneous spectra. A static
  mask applies the same frequency weighting regardless of the actual
  frequency content. For example, a window where the 12h harmonic is strong
  may need different treatment than one where it's weak.

  By conditioning the mask on the magnitude spectrum of the current input,
  the module can:
  - Enhance harmonics that are present in the current window
  - Suppress frequencies that are noise-dominated in this window
  - Adapt to non-stationary frequency content

Key properties:
  - O(C × N_freq + N_freq²) parameters (adaptation MLP shared across C)
  - Zero-init: starts as identity via res_scale=0
  - Adaptation network is shared across channels (keeps O(C) scaling)
  - The adaptation MLP takes mean magnitude across channels as input

Layer count controlled via --freq_v3_layers (default=0 = disabled).
"""

import torch
import torch.nn as nn


class FrequencyFilterV3Layer(nn.Module):
    """
    Adaptive frequency mask: static base × input-dependent modulation.

    Input:  [B, C, L]
    Output: [B, C, L]  (zero-init residual: x + res_scale * filtered)

    Architecture:
        x → rfft → X [B, C, N_freq]
               → mag = mean(|X|, dim=C) → MLP → mod [B, N_freq]
               → mask = sigmoid(W_static[C,N_freq]) × mod
               → X_filtered = X × mask
               → irfft → filtered → x + res_scale × filtered

    Parameters:
      - Static: C × N_freq
      - Adaptation MLP: N_freq → N_freq/2 → N_freq (shared across C)
    """

    def __init__(self, seq_len, channel_size):
        super().__init__()
        self.seq_len = seq_len
        self.n_freq = seq_len // 2 + 1
        self.channel_size = channel_size

        # Static per-channel base mask
        self.freq_weights = nn.Parameter(torch.zeros(channel_size, self.n_freq))

        # Adaptive modulation: magnitude spectrum → per-frequency modulation
        # Shared across channels (takes mean magnitude)
        self.adapt = nn.Sequential(
            nn.Linear(self.n_freq, max(self.n_freq // 2, 8)),
            nn.ReLU(),
            nn.Linear(max(self.n_freq // 2, 8), self.n_freq),
        )

        # Zero-init
        self.res_scale = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, L = x.shape

        # FFT
        X = torch.fft.rfft(x, dim=-1)              # [B, C, N_freq] complex

        # Mean magnitude across channels as input to adaptation MLP
        mag = torch.abs(X).mean(dim=1)              # [B, N_freq]

        # Adaptive modulation: sigmoid outputs ∈ (0,1)
        mod = torch.sigmoid(self.adapt(mag))         # [B, N_freq]

        # Static base mask
        static = torch.sigmoid(self.freq_weights)    # [C, N_freq]

        # Combine: static × adaptive
        mask = static.unsqueeze(0) * mod.unsqueeze(1) # [B, C, N_freq]

        # Apply
        X_filtered = X * mask

        # IFFT
        out = torch.fft.irfft(X_filtered, n=L, dim=-1)  # [B, C, L]

        return x + self.res_scale * out
