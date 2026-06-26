# CycleNet + MRT — Model Architecture

This repository contains the core model architecture code from the CycleNet + MRT time series forecasting project.

## Overview

CycleNet is a long-term time series forecasting model based on **channel-independent (CI) MLP** with learned cycle decomposition. This repo extends it with:

- **MRT (MultiResidualTrend)** — multi-resolution residual trend block
- **FrequencyFilter (V1–V4)** — frequency-domain spectral filters
- **SGF (Stability-Guided Frequency Gate)** — prior-initialized frequency gate
- **LowRankCycle** — low-rank reparameterization of cycle decomposition

## Files

```
models/
  CycleNet.py                    # Main model (RecurrentCycle + enhancement modules + CI MLP)
layers/
  MultiResidualTrend.py          # MRT: multi-scale avg_pool → conv → upsample fusion
  FrequencyFilter.py             # V1: static sigmoid magnitude mask (recommended)
  FrequencyFilterV2.py           # V2: complex weights (magnitude + phase)
  FrequencyFilterV3.py           # V3: input-adaptive dynamic mask
  FrequencyFilterV4.py           # V4: frequency band basis decomposition
  StabilityGuidedFrequencyGate.py  # SGF: frequency gate with stability-prior initialization
  LowRankCycle.py                # Low-rank U·V decomposition of cycle lookup table
```

## Key Design Principles

1. **Zero-init residual** — every module starts as identity (`res_scale=0`), ensuring training begins from the CycleNet baseline
2. **O(C) parameters** — all modules scale linearly with channel count, preventing overfitting on high-channel datasets (Traffic: 862ch)
3. **Layer count control** — each module controlled by `--xxx_layers N` (default=0 = disabled)
4. **No LayerNorm/Dropout** — preserves identity property at initialization
5. **Modular fusion** — supports serial (default) and parallel (delta-only) fusion modes

## Usage

```python
from models.CycleNet import Model

# configs should have attributes:
#   seq_len, pred_len, enc_in, cycle, model_type, d_model, use_revin
#   mrt_layers, freq_layers, freq_v2_layers, freq_v3_layers, freq_v4_layers, sgf_layers
#   fusion_mode ('serial' or 'parallel')
model = Model(configs)

# Forward: x [B, seq_len, enc_in], cycle_index [B]
y = model(x, cycle_index)  # [B, pred_len, enc_in]
```

## Citation

This code is part of the CycleNet + MRT project. See the full repository at [wakee111/TimeSeries](https://github.com/wakee111/TimeSeries) for training scripts, experiments, and logs.
