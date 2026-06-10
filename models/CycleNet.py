import torch
import torch.nn as nn
from layers.MultiResidualTrend import MultiResidualTrendLayer
from layers.FrequencyFilter import FrequencyFilterLayer
from layers.FrequencyFilterV2 import FrequencyFilterV2Layer
from layers.FrequencyFilterV3 import FrequencyFilterV3Layer
from layers.FrequencyFilterV4 import FrequencyFilterV4Layer


class RecurrentCycle(torch.nn.Module):
    # Thanks for the contribution of wayhoww.
    # The new implementation uses index arithmetic with modulo to directly gather cyclic data in a single operation,
    # while the original implementation manually rolls and repeats the data through looping.
    # It achieves a significant speed improvement (2x ~ 3x acceleration).
    # See https://github.com/ACAT-SCUT/CycleNet/pull/4 for more details.
    def __init__(self, cycle_len, channel_size):
        super(RecurrentCycle, self).__init__()
        self.cycle_len = cycle_len
        self.channel_size = channel_size
        self.data = torch.nn.Parameter(torch.zeros(cycle_len, channel_size), requires_grad=True)

    def forward(self, index, length):
        gather_index = (index.view(-1, 1) + torch.arange(length, device=index.device).view(1, -1)) % self.cycle_len
        return self.data[gather_index]


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.cycle_len = configs.cycle
        self.model_type = configs.model_type
        self.d_model = configs.d_model
        self.use_revin = configs.use_revin

        # Optional layer counts (0 = disabled)
        self.mrt_layers = getattr(configs, 'mrt_layers', 0)
        self.freq_layers = getattr(configs, 'freq_layers', 0)
        self.freq_v2_layers = getattr(configs, 'freq_v2_layers', 0)
        self.freq_v3_layers = getattr(configs, 'freq_v3_layers', 0)
        self.freq_v4_layers = getattr(configs, 'freq_v4_layers', 0)

        self.cycleQueue = RecurrentCycle(cycle_len=self.cycle_len, channel_size=self.enc_in)

        # MRT: Multi-Resolution Residual Trend blocks (low-frequency trends)
        self.mrt_blocks = nn.ModuleList([
            MultiResidualTrendLayer(
                seq_len=self.seq_len,
                channel_size=self.enc_in,
            )
            for _ in range(self.mrt_layers)
        ])

        # FreqFilter: Frequency-domain spectral filter (mid-frequency sub-harmonics)
        self.freq_blocks = nn.ModuleList([
            FrequencyFilterLayer(
                seq_len=self.seq_len,
                channel_size=self.enc_in,
            )
            for _ in range(self.freq_layers)
        ])

        # FreqFilterV2: Complex-weight spectral filter (magnitude + phase)
        self.freq_v2_blocks = nn.ModuleList([
            FrequencyFilterV2Layer(
                seq_len=self.seq_len,
                channel_size=self.enc_in,
            )
            for _ in range(self.freq_v2_layers)
        ])

        # FreqFilterV3: Adaptive spectral filter (input-dependent mask)
        self.freq_v3_blocks = nn.ModuleList([
            FrequencyFilterV3Layer(
                seq_len=self.seq_len,
                channel_size=self.enc_in,
            )
            for _ in range(self.freq_v3_layers)
        ])

        # FreqFilterV4: Band-basis spectral filter (K basis × per-channel mix)
        self.freq_v4_blocks = nn.ModuleList([
            FrequencyFilterV4Layer(
                seq_len=self.seq_len,
                channel_size=self.enc_in,
            )
            for _ in range(self.freq_v4_layers)
        ])

        # ---- Forecasting backbone ----
        assert self.model_type in ['linear', 'mlp']
        if self.model_type == 'linear':
            self.model = nn.Linear(self.seq_len, self.pred_len)
        elif self.model_type == 'mlp':
            self.model = nn.Sequential(
                nn.Linear(self.seq_len, self.d_model),
                nn.ReLU(),
                nn.Linear(self.d_model, self.pred_len)
            )

    def forward(self, x, cycle_index):
        # x: (batch_size, seq_len, enc_in), cycle_index: (batch_size,)

        # instance norm
        if self.use_revin:
            seq_mean = torch.mean(x, dim=1, keepdim=True)
            seq_var = torch.var(x, dim=1, keepdim=True) + 1e-5
            x = (x - seq_mean) / torch.sqrt(seq_var)

        # remove the cycle of the input data (learned seasonal-trend decomposition)
        x = x - self.cycleQueue(cycle_index, self.seq_len)

        # Permute to [B, C, L] for enhancement modules
        x = x.permute(0, 2, 1)  # [B, L, C] → [B, C, L]

        # MRT: multi-resolution trend extraction on residual (low-frequency)
        for mrt in self.mrt_blocks:
            x = mrt(x)

        # FreqFilter: frequency-domain spectral filtering (sub-harmonics)
        for freq in self.freq_blocks:
            x = freq(x)

        # FreqFilterV2: complex-weight spectral filtering (magnitude + phase)
        for freq_v2 in self.freq_v2_blocks:
            x = freq_v2(x)

        # FreqFilterV3: adaptive spectral filtering (input-dependent mask)
        for freq_v3 in self.freq_v3_blocks:
            x = freq_v3(x)

        # FreqFilterV4: band-basis spectral filtering
        for freq_v4 in self.freq_v4_blocks:
            x = freq_v4(x)

        # forecasting with channel independence (parameters-sharing)
        y = self.model(x).permute(0, 2, 1)

        # add back the cycle of the output data
        y = y + self.cycleQueue((cycle_index + self.seq_len) % self.cycle_len, self.pred_len)

        # instance denorm
        if self.use_revin:
            y = y * torch.sqrt(seq_var) + seq_mean

        return y
