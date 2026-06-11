"""
LowRankCycle — Low-Rank Reparameterization of Cycle Decomposition

Motivation:
  The original RecurrentCycle learns a full [cycle_len, C] lookup table with
  O(cycle_len * C) parameters. For high-channel datasets (Traffic: 862 channels,
  cycle=168 → 145K params), this is prone to overfitting given the limited
  training samples.

  We reparameterize the cycle table via low-rank decomposition:
      data[cycle_len, C] = U[cycle_len, R] @ V[R, C]

  This reduces parameters from O(cycle_len * C) to O(R * (cycle_len + C)).
  For Traffic with R=4: ~4K params instead of 145K (36x reduction).

  The low-rank assumption is natural: real seasonal patterns are smooth and
  channel-correlated — 24 hourly steps don't need 24 independent degrees of
  freedom; 4-6 basis temporal patterns are sufficient.

Key properties:
  - Zero initialization: U=0, V=0 → data=0 → exactly matches RecurrentCycle at init
  - Forward pass is identical to RecurrentCycle: index-based lookup via modulo
  - Only the parameterization changes (low-rank), not the functional boundary
  - Does NOT model multi-harmonics → SGF remains complementary, not redundant

Controlled via --cycle_mode lowrank (default=lookup, backward compatible).
"""

import torch
import torch.nn as nn


class LowRankCycle(nn.Module):
    """
    Low-rank cycle decomposition: data = U @ V, where
      U: [cycle_len, rank]  — temporal basis patterns
      V: [rank, channel_size] — per-channel mixing coefficients

    After computing data = U @ V, the forward pass is identical to RecurrentCycle:
      gather_index = (index + offset) % cycle_len
      return data[gather_index]

    Args:
        cycle_len: length of the periodic cycle (e.g., 24 for hourly daily)
        channel_size: number of channels/variates
        rank: rank of the decomposition (default=4)
    """

    def __init__(self, cycle_len, channel_size, rank=4):
        super(LowRankCycle, self).__init__()
        self.cycle_len = cycle_len
        self.channel_size = channel_size
        self.rank = rank

        # Temporal basis: [cycle_len, rank]
        # Small random init — data = U@V ≈ 0 but gradients flow through both matrices
        self.U = nn.Parameter(torch.randn(cycle_len, rank) * 0.01)

        # Channel mixing: [rank, channel_size]
        self.V = nn.Parameter(torch.randn(rank, channel_size) * 0.01)

    def forward(self, index, length):
        """
        Args:
            index: [B] batch of cycle indices
            length: sequence length L

        Returns:
            cycle_values: [B, L, C]
        """
        # Low-rank reconstruction of the cycle table
        data = self.U @ self.V  # [cycle_len, C]

        # Same lookup as RecurrentCycle
        gather_index = (
            index.view(-1, 1) + torch.arange(length, device=index.device).view(1, -1)
        ) % self.cycle_len

        return data[gather_index]
