"""
ETTh1 残差频谱分析
目的：查看 CycleNet 去除周期分量后，残差中是否还有次周期频率峰值
"""
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, '/data/data_huaji/timeSeries/CycleNetBaseLine')
from models.CycleNet import Model, RecurrentCycle
from data_provider.data_factory import data_provider

# 加载数据
class DataArgs:
    seq_len=96; label_len=0; pred_len=96; data='ETTh1'
    root_path='./dataset/ETT-small/'; data_path='ETTh1.csv'
    features='M'; target='OT'; freq='h'; batch_size=1; num_workers=0
    embed='timeF'; cycle=24

data_args = DataArgs()
test_set, test_loader = data_provider(data_args, 'test')

# 加载模型
class ModelArgs:
    seq_len=96; pred_len=96; enc_in=7; cycle=24
    model_type='mlp'; d_model=512; use_revin=1; mrt_layers=0

device = torch.device('cuda:0')
model = Model(ModelArgs()).to(device)
ckpt = '/data/data_huaji/timeSeries/CycleNet/checkpoints/ETTh1_96_96_CycleNet_ETTh1_ftM_sl96_pl96_cycle24_mlp_seed2024/checkpoint.pth'
model.load_state_dict(torch.load(ckpt, map_location=device), strict=False)
model.eval()

# 收集残差
all_residuals = []
all_originals = []
all_cycles = []

with torch.no_grad():
    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle) in enumerate(test_loader):
        if i >= 300: break
        batch_x = batch_x.float().to(device)  # [1, 96, 7]
        batch_cycle = batch_cycle.int().to(device)

        # RevIN if enabled
        x_norm = batch_x
        if model.use_revin:
            seq_mean = torch.mean(batch_x, dim=1, keepdim=True)
            seq_var = torch.var(batch_x, dim=1, keepdim=True) + 1e-5
            x_norm = (batch_x - seq_mean) / torch.sqrt(seq_var)

        cycle = model.cycleQueue(batch_cycle, model.seq_len)  # [1, 96, 7]
        residual = x_norm - cycle

        all_residuals.append(residual.cpu().numpy())
        all_originals.append(x_norm.cpu().numpy())
        all_cycles.append(cycle.cpu().numpy())

residuals = np.concatenate(all_residuals, axis=0)  # [300, 96, 7]
originals = np.concatenate(all_originals, axis=0)
cycles = np.concatenate(all_cycles, axis=0)
print(f"Collected {residuals.shape[0]} windows, shape: {residuals.shape}")

# FFT
N_fft = 96
freqs = np.fft.rfftfreq(N_fft, d=1.0)  # 1 sample = 1 hour

channel_names = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']

print("\n" + "="*70)
print("残差频谱峰值分析 (Residual Spectrum Peak Analysis)")
print("CycleNet cycle=24h (基频 f₀=1/24≈0.0417)")
print("="*70)

key_periods_h = [6, 8, 12, 24, 84, 168]  # quarter-day, third-day, half-day, day, half-week, week

for ch_idx, ch_name in enumerate(channel_names):
    res_fft = np.abs(np.fft.rfft(residuals[:, :, ch_idx], axis=1)).mean(axis=0)
    orig_fft = np.abs(np.fft.rfft(originals[:, :, ch_idx], axis=1)).mean(axis=0)

    # Energy ratio: residual vs original energy per frequency
    res_energy = res_fft[1:] ** 2
    orig_energy = orig_fft[1:] ** 2
    energy_ratio = res_energy / (orig_energy + 1e-10)

    print(f"\n--- {ch_name} ---")
    print(f"  Residual total energy: {res_energy.sum():.4f}")
    print(f"  Original total energy: {orig_energy.sum():.4f}")
    print(f"  Energy remaining in residual: {res_energy.sum()/orig_energy.sum()*100:.1f}%")

    for target_h in key_periods_h:
        f_target = 1.0 / target_h
        # Find nearest frequency bin
        idx = np.argmin(np.abs(freqs[1:] - f_target)) + 1
        ratio = energy_ratio[idx-1]
        orig_mag = orig_fft[idx]
        res_mag = res_fft[idx]
        marker = ""
        if ratio > 0.5:
            marker = " ⬅ HIGH residual energy!"
        if target_h == 24:
            marker += " [base cycle - should be mostly removed]"
        elif target_h == 12:
            marker += " [half-day]"
        elif target_h == 168:
            marker += " [weekly]"
        print(f"  Period={target_h:4d}h (f={f_target:.4f}): orig_mag={orig_mag:.4f}, res_mag={res_mag:.4f}, residual/orig ratio={ratio:.3f}{marker}")

    # Find top-5 peaks in residual
    print(f"  Top-5 residual frequency peaks:")
    peak_candidates = []
    for i in range(2, len(freqs)-1):
        if res_fft[i] > res_fft[i-1] and res_fft[i] > res_fft[i+1]:
            peak_candidates.append((1.0/freqs[i], res_fft[i]))
    peak_candidates.sort(key=lambda x: x[1], reverse=True)
    for period_h, mag in peak_candidates[:5]:
        print(f"    Period={period_h:.0f}h  f={1/period_h:.4f}  mag={mag:.4f}")

# ---- Aggregate plot for all channels ----
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
for ch_idx in range(7):
    ax = axes[ch_idx // 4, ch_idx % 4]
    res_fft = np.abs(np.fft.rfft(residuals[:, :, ch_idx], axis=1)).mean(axis=0)
    orig_fft = np.abs(np.fft.rfft(originals[:, :, ch_idx], axis=1)).mean(axis=0)
    ax.plot(freqs[1:], orig_fft[1:]/orig_fft[1:].max(), 'b-', alpha=0.4, linewidth=0.8, label='Original')
    ax.plot(freqs[1:], res_fft[1:]/res_fft[1:].max(), 'r-', alpha=0.8, linewidth=0.8, label='Residual')
    ax.set_title(channel_names[ch_idx])
    ax.set_xlim(0, 0.15)
    for h, ls, c, lb in [(24, '--', 'gray', '24h'), (12, '--', 'orange', '12h'), (168, '-.', 'green', '168h')]:
        ax.axvline(x=1/h, linestyle=ls, color=c, alpha=0.3)
    ax.legend(fontsize=7)

# 8th subplot: average across channels
ax = axes[1, 3]
avg_orig = np.zeros(len(freqs))
avg_res = np.zeros(len(freqs))
for ch_idx in range(7):
    avg_orig += np.abs(np.fft.rfft(originals[:, :, ch_idx], axis=1)).mean(axis=0)
    avg_res += np.abs(np.fft.rfft(residuals[:, :, ch_idx], axis=1)).mean(axis=0)
avg_orig /= 7; avg_res /= 7
ax.plot(freqs[1:], avg_orig[1:]/avg_orig[1:].max(), 'b-', alpha=0.4, linewidth=0.8, label='Original')
ax.plot(freqs[1:], avg_res[1:]/avg_res[1:].max(), 'r-', alpha=0.8, linewidth=0.8, label='Residual')
ax.set_title('Average (7 channels)')
ax.set_xlim(0, 0.15)
for h, ls, c, lb in [(24, '--', 'gray', '24h'), (12, '--', 'orange', '12h'), (168, '-.', 'green', '168h')]:
    ax.axvline(x=1/h, linestyle=ls, color=c, alpha=0.3)
ax.legend(fontsize=7)

plt.suptitle('ETTh1: Original vs Residual Spectrum (normalized)', fontsize=14)
plt.tight_layout()
os.makedirs('figs', exist_ok=True)
plt.savefig('figs/etth1_residual_spectrum.png', dpi=150)
print("\nSaved figs/etth1_residual_spectrum.png")
print("\nDone!")
