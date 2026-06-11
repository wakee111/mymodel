#!/usr/bin/env python3
"""
Compute frequency stability prior for SGF module.

For a given dataset, reads raw training data, estimates and removes the
base cycle component (via moving average), then computes per-frequency
stability S[k] = mean(|FFT(residual)[k]|) / (std(|FFT(residual)[k]|) + ε).

Usage:
  python scripts/compute_stability_prior.py \
      --data_path ./dataset/ETT-small/ETTh1.csv \
      --data ETTh1 \
      --cycle 24 --seq_len 96
"""

import argparse
import numpy as np
import pandas as pd
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_raw_data(root_path, data_path):
    """Load raw CSV data, return (data_array, num_features)."""
    df = pd.read_csv(os.path.join(root_path, data_path))
    # ETT datasets: first column is date, rest are features
    if 'date' in df.columns:
        df = df.drop(columns=['date'])
    return df.values.astype(np.float32), df.shape[1]


def naive_cycle_estimate(data, cycle_len):
    """
    Estimate the base cycle component using a moving average
    with window = cycle_len. This serves as a rough proxy for
    what RecurrentCycle will learn (f₀ component).

    Returns: cycle_estimate with same shape as data
    """
    from scipy.ndimage import uniform_filter1d
    estimate = np.zeros_like(data)
    for c in range(data.shape[1]):
        estimate[:, c] = uniform_filter1d(data[:, c], size=cycle_len, mode='reflect')
    return estimate


def compute_stability(residual, seq_len):
    """
    Compute per-frequency stability from residual signal.

    residual: [T, C] raw residual (data - naive cycle estimate)
    seq_len: sliding window length for FFT

    Returns: S[N_freq] stability scores
    """
    T, C = residual.shape
    n_freq = seq_len // 2 + 1

    # Sliding windows
    amplitudes = []
    for i in range(0, T - seq_len, seq_len // 2):  # 50% overlap
        window = residual[i:i + seq_len, :]  # [seq_len, C]
        if window.shape[0] < seq_len:
            break
        X = np.fft.rfft(window, axis=0)  # [N_freq, C]
        amplitudes.append(np.abs(X))

    # Stack: [N_windows, N_freq, C]
    amp_stack = np.stack(amplitudes, axis=0)

    # Mean amplitude across channels and windows: [N_freq]
    per_freq_mean = amp_stack.mean(axis=(0, 2))
    per_freq_std = amp_stack.std(axis=(0, 2))

    # Stability = CV^(-1): high mean / low std = stable
    stability = per_freq_mean / (per_freq_std + 1e-8)

    # Normalize to [0.01, 0.99] range for sigmoid later
    s_min, s_max = stability.min(), stability.max()
    if s_max - s_min > 1e-8:
        stability_norm = 0.01 + 0.98 * (stability - s_min) / (s_max - s_min)
    else:
        stability_norm = 0.5 * np.ones_like(stability)

    return stability_norm.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default='./dataset/ETT-small/')
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--data', type=str, required=True, help='Dataset name for output filename')
    parser.add_argument('--cycle', type=int, required=True)
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--output_dir', type=str, default='./stability_priors/')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load raw data
    data, C = load_raw_data(args.root_path, args.data_path)
    print(f"Loaded {args.data_path}: shape={data.shape}, channels={C}")

    # 2. Naive cycle estimate (moving average)
    cycle_est = naive_cycle_estimate(data, args.cycle)

    # 3. Residual = raw - cycle estimate
    residual = data - cycle_est
    print(f"Residual energy ratio: {np.var(residual) / np.var(data) * 100:.1f}%")

    # 4. Compute frequency stability
    stability = compute_stability(residual, args.seq_len)
    print(f"Stability prior: min={stability.min():.4f}, max={stability.max():.4f}, "
          f"mean={stability.mean():.4f}")

    # 5. Save
    output_path = os.path.join(args.output_dir, f'stability_{args.data}_cycle{args.cycle}_L{args.seq_len}.npy')
    np.save(output_path, stability)
    print(f"Saved to {output_path}")


if __name__ == '__main__':
    main()
