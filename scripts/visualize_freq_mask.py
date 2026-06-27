#!/usr/bin/env python
"""
Visualize learned FrequencyFilter mask from a trained checkpoint.

Usage:
    python scripts/visualize_freq_mask.py \
        --ckpt checkpoints/XXX/checkpoint.pth \
        --dataset solar --pred_len 720
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.CycleNet import Model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True, help='Path to checkpoint.pth')
    parser.add_argument('--dataset', type=str, default='solar',
                        choices=['solar', 'etth1', 'etth2', 'ettm1', 'ettm2', 'weather', 'electricity', 'traffic'])
    parser.add_argument('--pred_len', type=int, default=720)
    parser.add_argument('--save', type=str, default='', help='Save figure to path')
    parser.add_argument('--n_channels', type=int, default=6, help='Number of channels to plot')
    args = parser.parse_args()

    # Dataset config
    configs = {
        'solar':       {'enc_in': 137, 'cycle': 144, 'use_revin': 0, 'data': 'Solar'},
        'etth1':       {'enc_in': 7,   'cycle': 24,  'use_revin': 1, 'data': 'ETTh1'},
        'etth2':       {'enc_in': 7,   'cycle': 24,  'use_revin': 1, 'data': 'ETTh2'},
        'ettm1':       {'enc_in': 7,   'cycle': 96,  'use_revin': 1, 'data': 'ETTm1'},
        'ettm2':       {'enc_in': 7,   'cycle': 96,  'use_revin': 1, 'data': 'ETTm2'},
        'weather':     {'enc_in': 21,  'cycle': 144, 'use_revin': 1, 'data': 'custom'},
        'electricity': {'enc_in': 321, 'cycle': 168, 'use_revin': 1, 'data': 'custom'},
        'traffic':     {'enc_in': 862, 'cycle': 168, 'use_revin': 1, 'data': 'custom'},
    }
    cfg = configs[args.dataset]

    # Load checkpoint first to infer layer config
    device = torch.device('cpu')
    state = torch.load(args.ckpt, map_location='cpu')

    # Infer layer counts from checkpoint keys (handle both mrt_present and mrt_absent)
    has_mrt = any('mrt_blocks' in k for k in state.keys())
    has_freq = any('freq_blocks' in k for k in state.keys())
    has_freq_v2 = any('freq_v2_blocks' in k for k in state.keys())
    has_freq_v3 = any('freq_v3_blocks' in k for k in state.keys())
    has_freq_v4 = any('freq_v4_blocks' in k for k in state.keys())
    has_sgf = any('sgf_blocks' in k for k in state.keys())

    print(f'Inferred from checkpoint: MRT={has_mrt}, FreqV1={has_freq}, '
          f'FreqV2={has_freq_v2}, FreqV3={has_freq_v3}, FreqV4={has_freq_v4}, SGF={has_sgf}')

    if not has_freq:
        print('ERROR: This checkpoint has no freq_blocks. Cannot visualize.')
        sys.exit(1)

    # Create a fake args for model construction
    class FakeArgs:
        seq_len = 96
        pred_len = args.pred_len
        enc_in = cfg['enc_in']
        cycle = cfg['cycle']
        model_type = 'mlp'
        d_model = 512
        use_revin = cfg['use_revin']
        fusion_mode = 'serial'
        fusion_order = 'mrt_freq'
        fusion_gate = 0
        mrt_layers = 1 if has_mrt else 0
        freq_layers = 1 if has_freq else 0
        freq_v2_layers = 1 if has_freq_v2 else 0
        freq_v3_layers = 1 if has_freq_v3 else 0
        freq_v4_layers = 1 if has_freq_v4 else 0
        sgf_layers = 1 if has_sgf else 0
        sgf_prior = None
        cycle_mode = 'lookup'
        cycle_rank = 4

    model = Model(FakeArgs()).to(device)
    model.load_state_dict(state)

    # Extract freq_weights
    if len(model.freq_blocks) == 0:
        print('ERROR: This checkpoint has no freq_blocks (freq_layers=0)')
        sys.exit(1)

    freq_weights = model.freq_blocks[0].freq_weights.detach().cpu().numpy()  # [C, 49]
    mask = 1.0 / (1.0 + np.exp(-freq_weights))  # sigmoid, [C, 49]
    res_scale = model.freq_blocks[0].res_scale.item()

    C, N_freq = mask.shape
    print(f'Dataset: {args.dataset}, C={C}, N_freq={N_freq}')
    print(f'res_scale = {res_scale:.6f}')
    print(f'Mask stats: min={mask.min():.4f}, max={mask.max():.4f}, mean={mask.mean():.4f}')

    # Frequency labels
    seq_len = 96
    freqs = np.arange(N_freq) / seq_len          # cycles per step
    periods = np.where(freqs > 0, 1.0 / freqs, np.inf)  # steps per cycle

    # Plot
    n_cols = min(args.n_channels, C)
    fig, axes = plt.subplots(n_cols, 1, figsize=(14, 2.5 * n_cols), sharex=True)
    if n_cols == 1:
        axes = [axes]

    # Color-code by frequency band
    # DC: bin 0, Very low: bin 1-2, Low: bin 3-5, Mid: bin 6-18, High: bin 19-48
    band_colors = []
    for b in range(N_freq):
        if b == 0:
            band_colors.append('#333333')       # DC - dark gray
        elif b <= 2:
            band_colors.append('#1f77b4')        # very low - blue
        elif b <= 5:
            band_colors.append('#2ca02c')        # low - green
        elif b <= 18:
            band_colors.append('#ff7f0e')        # mid - orange (target band)
        else:
            band_colors.append('#d62728')        # high - red (noise)

    for i in range(n_cols):
        ax = axes[i]
        ch = i  # show first N channels

        for b in range(N_freq):
            ax.bar(b, mask[ch, b], color=band_colors[b], width=0.8, alpha=0.85)

        ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=0.8, label='init=0.5 (neutral)')
        ax.set_ylabel(f'Ch {ch}\nmask', fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.set_xlim(-1, N_freq)
        ax.legend(loc='upper right', fontsize=8)

        # Add secondary x-axis with periods on top
        ax2 = ax.twiny()
        tick_bins = [0, 6, 12, 18, 24, 30, 36, 42, 48]
        tick_labels = []
        for b in tick_bins:
            if b == 0:
                tick_labels.append('DC')
            else:
                p = periods[b]
                tick_labels.append(f'{p:.0f}步')
        ax2.set_xticks(tick_bins)
        ax2.set_xticklabels(tick_labels, fontsize=8)
        ax2.set_xlim(-1, N_freq)

    axes[-1].set_xlabel('Frequency bin (0=DC, 1~48=1/96~48/96 cycles/step)', fontsize=11)

    # Legend for bands
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#333333', label='Bin 0: DC'),
        Patch(facecolor='#1f77b4', label='Bin 1-2: 极低频 (48-96步)'),
        Patch(facecolor='#2ca02c', label='Bin 3-5: 低频 (19-32步)'),
        Patch(facecolor='#ff7f0e', label='Bin 6-18: 中频 (5-16步) ← Freq 目标'),
        Patch(facecolor='#d62728', label='Bin 19-48: 高频 (2-5步) → 噪声'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=5, fontsize=8,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(f'FrequencyFilter Learned Mask — {args.dataset.upper()} pred={args.pred_len}\n'
                 f'(sigmoid(freq_weights), res_scale={res_scale:.4f})',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()

    if args.save:
        plt.savefig(args.save, dpi=150, bbox_inches='tight')
        print(f'Saved to {args.save}')
    else:
        plt.show()


if __name__ == '__main__':
    main()
