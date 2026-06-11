"""
StabilityGuidedFrequencyGate (SGF) — Prior-Initialized Spectral Filter

Motivation:
  FrequencyFilter V1 uses sigmoid(0)=0.5 initialization for all frequency bins.
  Under RevIN (which uniformly scales all non-zero frequencies), the model
  cannot distinguish important vs unimportant frequencies from scratch.

  SGF addresses this by:
  1. Pre-computing frequency stability S[k] from raw training data residuals
     (BEFORE RevIN normalization — bypassing the RevIN bottleneck)
  2. Using S[k] as a bias prior for the sigmoid gate initialization

  S[k] = mean(|FFT(residual_raw)[k]|) / (std(|FFT(residual_raw)[k]|) + ε)

  High S[k] → frequency k is consistently present across the dataset
  Low S[k] → frequency k is noise-dominated or inconsistent

  The residual_raw is computed by removing a naive cycle estimate (moving avg)
  from the raw data, so S[k] specifically measures sub-harmonic stability,
  avoiding overlap with the RecurrentCycle module.

  At initialization:
    gate[k] = sigmoid(logit_prior[k] + bias[C,k])
    where logit_prior[k] = log(S[k] / (1 - S[k]))

  During training, bias[C,k] learns to adjust the gate per-channel.

Key properties:
  - O(C × N_freq) params (same as V1)
  - Zero-init residual via res_scale=0 (same as V1)
  - Prior computed from raw data → bypasses RevIN compression
  - Prior measured on cycle-removed residual → doesn't duplicate RecurrentCycle
  - No LayerNorm/Dropout, preserving identity at initialization

Inspired by:
  - FredNormer (ICLR 2025): frequency stability measure for normalization
  - Our key insight: compute stability BEFORE RevIN, apply gate AFTER RevIN

Controlled via --sgf_layers (default=0 = disabled).
"""

import torch
import torch.nn as nn
import numpy as np
import os


class StabilityGuidedFrequencyGateLayer(nn.Module):
    """
    Frequency gate with stability-prior initialization.

    Input:  [B, C, L]
    Output: [B, C, L]  (zero-init residual: x + res_scale * filtered)

    Architecture:
        x → rfft → X [B, C, N_freq]
               → gate = sigmoid(logit_prior[N_freq] + bias[C,N_freq])
               → X_filtered = X × gate
               → irfft → filtered
               → x + res_scale × filtered

    Parameters: C × N_freq + 1 (same as V1)
    """

    def __init__(self, seq_len, channel_size, stability_prior=None):
        """
        Args:
            seq_len: input sequence length (L)
            channel_size: number of channels (C)
            stability_prior: [N_freq] numpy array or torch tensor.
                            If None, falls back to V1 behavior (zero init bias).
        """
        super().__init__()
        self.seq_len = seq_len
        self.n_freq = seq_len // 2 + 1
        self.channel_size = channel_size

        # 1. Build logit prior from stability scores
        if stability_prior is not None:
            if isinstance(stability_prior, np.ndarray):
                stability_prior = torch.from_numpy(stability_prior).float()
            # Clamp to valid sigmoid range
            S = torch.clamp(stability_prior, 0.01, 0.99)
            logit_prior = torch.log(S / (1.0 - S))  # [N_freq]
            self.register_buffer('logit_prior', logit_prior)
            self.has_prior = True
        else:
            # Fallback: no prior = V1 behavior
            self.register_buffer('logit_prior', torch.zeros(self.n_freq))
            self.has_prior = False

        # 2. Per-channel learnable bias (initialized to 0)
        # sigmoid(prior + 0) → starts at the prior value
        self.freq_bias = nn.Parameter(torch.zeros(channel_size, self.n_freq))

        # 3. Zero-init residual scale
        self.res_scale = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, L = x.shape

        # FFT
        X = torch.fft.rfft(x, dim=-1)  # [B, C, N_freq]

        # Gate: sigmoid(prior + learnable bias)
        # Prior is shared across channels, bias is per-channel
        gate = torch.sigmoid(self.logit_prior.unsqueeze(0) + self.freq_bias)  # [C, N_freq]

        # Apply
        X_filtered = X * gate.unsqueeze(0)  # [B, C, N_freq]

        # IFFT
        out = torch.fft.irfft(X_filtered, n=L, dim=-1)  # [B, C, L]

        # Zero-init residual
        return x + self.res_scale * out

    def get_gate_profile(self):
        """Return the current gate values for inspection (without batch dim)."""
        with torch.no_grad():
            return torch.sigmoid(self.logit_prior.unsqueeze(0) + self.freq_bias)  # [C, N_freq]
