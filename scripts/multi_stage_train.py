#!/usr/bin/env python
"""
Generalized multi-stage training: MRT warmstart + Freq fine-tune + progressive unfreeze.

Supports: solar, etth1, etth2, ettm1, ettm2, weather, electricity, traffic

Stage 1:  Train MRT only (mrt_layers=1, freq_layers=0), save checkpoint.
Stage 2:  Load MRT checkpoint, add Freq branch, freeze backbone+MRT+cycle, train Freq 2-3 epochs.
Stage 3a: Freeze cycleQueue + mrt_blocks, train model + freq_blocks (progressive unfreeze:
          let MLP learn to consume Freq output before MRT competes for gradients).
Stage 3b: Unfreeze all, group-wise LR (freq high / model medium / mrt+cycle low),
          optional FreDF.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/multi_stage_train.py \\
        --dataset solar --pred_len 192 [--stage3_fredf] [--gpu 0]
"""

import os, sys, argparse, time, copy, random
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exp.exp_main import Exp_Main
from models.CycleNet import Model
from data_provider.data_factory import data_provider
from utils.tools import EarlyStopping, adjust_learning_rate

PYTHON = '/data/data_linsf/Anaconda3/envs/TimeSeries_huaji/bin/python'
ROOT = '/data/data_huaji/timeSeries/CycleNetBaseLine'

# ─── Dataset configurations ───────────────────────────────────────

DATASET_CONFIGS = {
    'etth1': {
        'data': 'ETTh1', 'root_path': './dataset/ETT-small/', 'data_path': 'ETTh1.csv',
        'enc_in': 7, 'cycle': 24, 'use_revin': 1, 'batch_size': 256, 'lr': 0.005,
        'freq': 'h',
    },
    'etth2': {
        'data': 'ETTh2', 'root_path': './dataset/ETT-small/', 'data_path': 'ETTh2.csv',
        'enc_in': 7, 'cycle': 24, 'use_revin': 1, 'batch_size': 256, 'lr': 0.005,
        'freq': 'h',
    },
    'ettm1': {
        'data': 'ETTm1', 'root_path': './dataset/ETT-small/', 'data_path': 'ETTm1.csv',
        'enc_in': 7, 'cycle': 96, 'use_revin': 1, 'batch_size': 256, 'lr': 0.005,
        'freq': 'h',
    },
    'ettm2': {
        'data': 'ETTm2', 'root_path': './dataset/ETT-small/', 'data_path': 'ETTm2.csv',
        'enc_in': 7, 'cycle': 96, 'use_revin': 1, 'batch_size': 256, 'lr': 0.005,
        'freq': 'h',
    },
    'electricity': {
        'data': 'custom', 'root_path': './dataset/electricity/', 'data_path': 'electricity.csv',
        'enc_in': 321, 'cycle': 168, 'use_revin': 1, 'batch_size': 64, 'lr': 0.005,
        'freq': 'h',
    },
    'traffic': {
        'data': 'custom', 'root_path': './dataset/traffic/', 'data_path': 'traffic.csv',
        'enc_in': 862, 'cycle': 168, 'use_revin': 1, 'batch_size': 64, 'lr': 0.002,
        'freq': 'h',
    },
    'weather': {
        'data': 'custom', 'root_path': './dataset/weather/', 'data_path': 'weather.csv',
        'enc_in': 21, 'cycle': 144, 'use_revin': 1, 'batch_size': 256, 'lr': 0.005,
        'freq': 'h',
    },
    'solar': {
        'data': 'Solar', 'root_path': './dataset/Solar/', 'data_path': 'solar_AL.txt',
        'enc_in': 137, 'cycle': 144, 'use_revin': 0, 'batch_size': 64, 'lr': 0.01,
        'freq': 'h',
    },
}

# ─── Args builder ─────────────────────────────────────────────────

def build_args(dataset, pred_len, seed=2024):
    """Build args namespace for a given dataset."""
    cfg = DATASET_CONFIGS[dataset]

    parser = argparse.ArgumentParser()
    # Required
    parser.add_argument('--is_training', type=int, default=1)
    parser.add_argument('--model_id', type=str, default='{}_96_{}'.format(dataset.upper(), pred_len))
    parser.add_argument('--model', type=str, default='CycleNet')
    parser.add_argument('--data', type=str, default=cfg['data'])
    # Data
    parser.add_argument('--root_path', type=str, default=cfg['root_path'])
    parser.add_argument('--data_path', type=str, default=cfg['data_path'])
    parser.add_argument('--features', type=str, default='M')
    parser.add_argument('--target', type=str, default='OT')
    parser.add_argument('--freq', type=str, default=cfg['freq'])
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/')
    # Task
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--label_len', type=int, default=0)
    parser.add_argument('--pred_len', type=int, default=pred_len)
    # CycleNet
    parser.add_argument('--cycle', type=int, default=cfg['cycle'])
    parser.add_argument('--cycle_mode', type=str, default='lookup')
    parser.add_argument('--cycle_rank', type=int, default=4)
    parser.add_argument('--model_type', type=str, default='mlp')
    parser.add_argument('--use_revin', type=int, default=cfg['use_revin'])
    # Enhancement modules
    parser.add_argument('--mrt_layers', type=int, default=1)
    parser.add_argument('--freq_layers', type=int, default=0)
    parser.add_argument('--freq_v2_layers', type=int, default=0)
    parser.add_argument('--freq_v3_layers', type=int, default=0)
    parser.add_argument('--freq_v4_layers', type=int, default=0)
    parser.add_argument('--sgf_layers', type=int, default=0)
    parser.add_argument('--sgf_prior_path', type=str, default='')
    parser.add_argument('--freq_loss_alpha', type=float, default=1.0)
    parser.add_argument('--fusion_mode', type=str, default='serial')
    parser.add_argument('--fusion_order', type=str, default='mrt_freq')
    parser.add_argument('--fusion_gate', type=int, default=0)
    # Model
    parser.add_argument('--enc_in', type=int, default=cfg['enc_in'])
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--fc_dropout', type=float, default=0.05)
    parser.add_argument('--head_dropout', type=float, default=0.0)
    # Optim
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--itr', type=int, default=1)
    parser.add_argument('--train_epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=cfg['batch_size'])
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--learning_rate', type=float, default=cfg['lr'])
    parser.add_argument('--des', type=str, default='test')
    parser.add_argument('--loss', type=str, default='mse')
    parser.add_argument('--lradj', type=str, default='type3')
    parser.add_argument('--pct_start', type=float, default=0.3)
    parser.add_argument('--use_amp', type=bool, default=False)
    # Dummy args
    parser.add_argument('--patch_len', type=int, default=16)
    parser.add_argument('--stride', type=int, default=8)
    parser.add_argument('--padding_patch', type=str, default='end')
    parser.add_argument('--revin', type=int, default=0)
    parser.add_argument('--affine', type=int, default=0)
    parser.add_argument('--subtract_last', type=int, default=0)
    parser.add_argument('--decomposition', type=int, default=0)
    parser.add_argument('--kernel_size', type=int, default=25)
    parser.add_argument('--individual', type=int, default=0)
    parser.add_argument('--rnn_type', type=str, default='gru')
    parser.add_argument('--dec_way', type=str, default='pmf')
    parser.add_argument('--seg_len', type=int, default=48)
    parser.add_argument('--channel_id', type=int, default=1)
    parser.add_argument('--period_len', type=int, default=24)
    parser.add_argument('--embed_type', type=int, default=0)
    parser.add_argument('--dec_in', type=int, default=7)
    parser.add_argument('--c_out', type=int, default=7)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--e_layers', type=int, default=2)
    parser.add_argument('--d_layers', type=int, default=1)
    parser.add_argument('--d_ff', type=int, default=2048)
    parser.add_argument('--moving_avg', type=int, default=25)
    parser.add_argument('--factor', type=int, default=1)
    parser.add_argument('--distil', type=bool, default=True)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--embed', type=str, default='timeF')
    parser.add_argument('--activation', type=str, default='gelu')
    parser.add_argument('--output_attention', type=bool, default=False)
    parser.add_argument('--do_predict', type=bool, default=False)
    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--use_multi_gpu', type=bool, default=False)
    parser.add_argument('--test_flop', type=bool, default=False)

    args = parser.parse_args([])

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    return args


# ─── Data loaders ─────────────────────────────────────────────────

def get_data_loaders(args):
    train_set, train_loader = data_provider(args, 'train')
    val_set, val_loader = data_provider(args, 'val')
    test_set, test_loader = data_provider(args, 'test')
    return train_loader, val_loader, test_loader


# ─── Setting builder ──────────────────────────────────────────────

def make_setting(args, stage, seed=2024):
    mrt_l = args.mrt_layers
    freq_l = args.freq_layers
    suffix = ''
    if mrt_l: suffix += '_mrt{}'.format(mrt_l)
    if freq_l: suffix += '_freq{}'.format(freq_l)
    if args.freq_loss_alpha < 1.0:
        suffix += '_fredf{}'.format(str(args.freq_loss_alpha).replace('.', ''))
    suffix += '_stage{}'.format(stage)
    return '{}_{}_{}_ft{}_sl{}_pl{}_cycle{}_{}{}_seed{}'.format(
        args.model_id, args.model, args.data, args.features,
        args.seq_len, args.pred_len, args.cycle, args.model_type, suffix, seed)


# ─── Training / validation / testing ──────────────────────────────

def train_one_epoch(model, train_loader, optimizer, criterion, device, args, scheduler=None, epoch=0):
    model.train()
    total_loss = []
    train_steps = len(train_loader)
    time_now = time.time()
    iter_count = 0

    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle) in enumerate(train_loader):
        iter_count += 1
        optimizer.zero_grad()
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        batch_cycle = batch_cycle.int().to(device)

        outputs = model(batch_x, batch_cycle)

        f_dim = -1 if args.features == 'MS' else 0
        outputs = outputs[:, -args.pred_len:, f_dim:]
        batch_y = batch_y[:, -args.pred_len:, f_dim:].to(device)

        loss = criterion(outputs, batch_y)

        if args.freq_loss_alpha < 1.0:
            loss_freq = (torch.fft.rfft(outputs, dim=1) - torch.fft.rfft(batch_y, dim=1)).abs().mean()
            loss = args.freq_loss_alpha * loss + (1 - args.freq_loss_alpha) * loss_freq

        # Sparsity regularizer: push freq mask away from 0.5
        sparsity_lambda = getattr(args, 'freq_sparsity_lambda', 0.0)
        if sparsity_lambda > 0 and hasattr(model, 'get_freq_sparsity_loss'):
            loss_sparsity = model.get_freq_sparsity_loss()
            loss = loss + sparsity_lambda * loss_sparsity

        loss.backward()
        optimizer.step()

        if args.lradj == 'TST' and scheduler:
            adjust_learning_rate(optimizer, scheduler, epoch + 1, args, printout=False)
            scheduler.step()

        total_loss.append(loss.item())

        if (i + 1) % 100 == 0:
            speed = (time.time() - time_now) / iter_count
            left = speed * ((args.train_epochs - epoch) * train_steps - i)
            print('    iters: {}, epoch: {} | loss: {:.7f} | speed: {:.4f}s/iter; left: {:.0f}s'.format(
                i + 1, epoch + 1, loss.item(), speed, left))
            iter_count = 0
            time_now = time.time()

    return np.average(total_loss)


def validate(model, val_loader, criterion, device, args):
    model.eval()
    total_loss = []
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle in val_loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_cycle = batch_cycle.int().to(device)

            outputs = model(batch_x, batch_cycle)

            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:].to(device)

            loss = criterion(outputs, batch_y)
            total_loss.append(loss.item())

        model.train()
        return np.average(total_loss)


def test_model(model, test_loader, device, args, setting):
    from utils.metrics import metric

    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle in test_loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_cycle = batch_cycle.int().to(device)

            outputs = model(batch_x, batch_cycle)
            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:]

            preds.append(outputs.cpu().numpy())
            trues.append(batch_y.cpu().numpy())

    preds = np.concatenate(preds, axis=0).reshape(-1, args.pred_len, args.enc_in)
    trues = np.concatenate(trues, axis=0).reshape(-1, args.pred_len, args.enc_in)

    mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
    mae, mse, rmse = float(mae), float(mse), float(rmse)
    mape, mspe = float(mape), float(mspe)
    rse, corr = float(np.mean(rse)), float(np.mean(corr))
    smape_val = float(200 * np.mean(np.abs(preds - trues) / (np.abs(preds) + np.abs(trues) + 1e-8)))

    # Save results
    folder = './test_results/{}/'.format(setting)
    os.makedirs(folder, exist_ok=True)
    np.save(folder + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe, rse, corr, smape_val]))
    np.save(folder + 'pred.npy', preds)
    np.save(folder + 'true.npy', trues)
    np.save(folder + 'residual.npy', preds - trues)

    return mse, mae, smape_val


def _diagnose_module_state(model, stage_label=""):
    """Print key diagnostic metrics for Freq/MRT/Gate module state."""
    parts = ["[diag {}]".format(stage_label)] if stage_label else ["[diag]"]

    # Freq branch diagnostics
    if hasattr(model, 'freq_blocks') and len(model.freq_blocks) > 0:
        fb = model.freq_blocks[0]
        rs = fb.res_scale.item()
        mask = torch.sigmoid(fb.freq_weights)
        parts.append("freq.rs={:+.4f} mask.mean={:.3f} mask.std={:.3f}".format(rs, mask.mean().item(), mask.std().item()))

    # MRT branch diagnostics
    if hasattr(model, 'mrt_blocks') and len(model.mrt_blocks) > 0:
        mb = model.mrt_blocks[0]
        parts.append("mrt.rs={:+.4f}".format(mb.res_scale.item()))

    # Fusion gate diagnostics (only when gate is active)
    if hasattr(model, 'gate_mrt_logit'):
        gm = torch.sigmoid(model.gate_mrt_logit).mean().item()
        gf = torch.sigmoid(model.gate_freq_logit).mean().item()
        parts.append("gate_mrt={:.4f} gate_freq={:.4f}".format(gm, gf))

    print("  ".join(parts))


# ─── Stage helpers ────────────────────────────────────────────────

def stage1_train_mrt(args, device):
    """Train MRT only, save checkpoint."""
    print('\n' + '='*60)
    print('STAGE 1: Train MRT only (freq_layers=0)')
    print('='*60)

    args.mrt_layers = 1
    args.freq_layers = 0
    args.freq_loss_alpha = 1.0  # pure MSE
    args.train_epochs = 30
    args.patience = 15

    setting = make_setting(args, 1)
    ckpt_dir = os.path.join(args.checkpoints, setting)
    os.makedirs(ckpt_dir, exist_ok=True)

    train_loader, val_loader, test_loader = get_data_loaders(args)

    model = Model(args).float().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = lr_scheduler.OneCycleLR(
        optimizer, steps_per_epoch=len(train_loader),
        pct_start=args.pct_start, epochs=args.train_epochs, max_lr=args.learning_rate)

    early_stopping = EarlyStopping(patience=args.patience, verbose=True)

    for epoch in range(args.train_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, args, scheduler, epoch)
        val_loss = validate(model, val_loader, criterion, device, args)
        test_loss = validate(model, test_loader, criterion, device, args)
        print('Epoch: {}, Steps: {} | Train: {:.7f} Vali: {:.7f} Test: {:.7f}'.format(
            epoch + 1, len(train_loader), train_loss, val_loss, test_loss))

        early_stopping(val_loss, model, ckpt_dir)
        if early_stopping.early_stop:
            print('Early stopping at epoch {}'.format(epoch + 1))
            break

        if args.lradj != 'TST':
            adjust_learning_rate(optimizer, scheduler, epoch + 1, args)

    best_path = os.path.join(ckpt_dir, 'checkpoint.pth')
    model.load_state_dict(torch.load(best_path))
    mse, mae, smape = test_model(model, test_loader, device, args, setting)
    print('STAGE 1 RESULT: mse={:.4f}, mae={:.4f}, smape={:.4f}'.format(mse, mae, smape))

    with open('result.txt', 'a') as f:
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, smape:{}\n\n'.format(mse, mae, smape))

    return best_path


# ─── Plan C: Separate MRT & Freq experts, then merge ──────────────

def stage2p_train_freq_only(args, device, stage1_ckpt):
    """Plan C Stage 2': Train Freq in MRT context (parallel+gate), full 30 epochs.

    Architecture: parallel fusion with frozen MRT
        x_base → MRT(frozen, S1 weights) → d_mrt
        x_base → Freq(trainable)          → d_freq
        x = x_base + d_mrt + d_freq       → MLP(frozen, S1 weights)

    Key idea: Freq gets almost all gradient, but learns in the CONTEXT of MRT.
    This is different from old Plan C (Freq-only, no MRT present): Freq now
    learns what spectral filtering complements MRT's trend extraction.
    Gate params are also trainable, initialized balanced at 0.5/0.5.
    """
    print('\n' + '='*60)
    print("STAGE 2': Train Freq in MRT context (parallel+gate, MRT+MLP frozen)")
    print('='*60)

    args.mrt_layers = 1   # MRT present but frozen
    args.freq_layers = 1
    args.fusion_mode = 'parallel'
    args.fusion_gate = 1
    args.freq_loss_alpha = 1.0  # pure MSE
    args.train_epochs = 30
    args.patience = 15

    setting = make_setting(args, '2p')
    ckpt_dir = os.path.join(args.checkpoints, setting)
    os.makedirs(ckpt_dir, exist_ok=True)

    train_loader, val_loader, test_loader = get_data_loaders(args)

    model = Model(args).float().to(device)

    # --- Override gate init: balanced experts ---
    model.gate_mrt_logit.data.fill_(0.0)
    model.gate_freq_logit.data.fill_(0.0)
    print('[PlanC Stage2\' gate] init gate_mrt ≈ 0.5000, gate_freq ≈ 0.5000 (balanced)')

    # --- Load MRT+MLP from Stage 1 ---
    s1_state = torch.load(stage1_ckpt)
    model_state = model.state_dict()
    loaded, fresh = 0, 0
    for k in model_state.keys():
        if k.startswith('freq_blocks') or k.startswith('gate_'):
            fresh += 1  # Freq + gate: fresh (trainable)
        elif k in s1_state and s1_state[k].shape == model_state[k].shape:
            model_state[k] = s1_state[k]
            loaded += 1
        else:
            fresh += 1
    model.load_state_dict(model_state)
    print('Loaded {} params from S1 (MRT+cycle+MLP), {} fresh (Freq+gate)'.format(loaded, fresh))

    # --- Freeze MRT + cycle + MLP, train only Freq + gate ---
    frozen_prefixes = ['cycleQueue', 'mrt_blocks', 'model']
    for name, param in model.named_parameters():
        if any(name.startswith(p) for p in frozen_prefixes):
            param.requires_grad = False

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print('Trainable: {:,} / {:,} ({:.1f}%) — Freq+gate only'.format(
        n_trainable, n_total, 100*n_trainable/n_total))

    criterion = nn.MSELoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    scheduler = lr_scheduler.OneCycleLR(
        optimizer, steps_per_epoch=len(train_loader),
        pct_start=args.pct_start, epochs=args.train_epochs, max_lr=args.learning_rate)

    early_stopping = EarlyStopping(patience=args.patience, verbose=True)

    for epoch in range(args.train_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, args, scheduler, epoch)
        val_loss = validate(model, val_loader, criterion, device, args)
        test_loss = validate(model, test_loader, criterion, device, args)
        print('Epoch: {}, Steps: {} | Train: {:.7f} Vali: {:.7f} Test: {:.7f}'.format(
            epoch + 1, len(train_loader), train_loss, val_loss, test_loss))
        _diagnose_module_state(model, "2'-{}".format(epoch + 1))

        early_stopping(val_loss, model, ckpt_dir)
        if early_stopping.early_stop:
            print('Early stopping at epoch {}'.format(epoch + 1))
            break

        if args.lradj != 'TST':
            adjust_learning_rate(optimizer, scheduler, epoch + 1, args)

    best_path = os.path.join(ckpt_dir, 'checkpoint.pth')
    model.load_state_dict(torch.load(best_path))
    mse, mae, smape = test_model(model, test_loader, device, args, setting)
    print("STAGE 2' RESULT: mse={:.4f}, mae={:.4f}, smape={:.4f}".format(mse, mae, smape))

    with open('result.txt', 'a') as f:
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, smape:{}\n\n'.format(mse, mae, smape))

    return best_path


def stage3p_merge_and_finetune(args, device, stage1_ckpt, stage2p_ckpt):
    """Plan C Stage 3: Merge MRT and Freq experts via PARALLEL fusion with gate.

    Architecture (parallel, NOT serial):
        x_base (raw residual)
          ├── MRT(x_base) → d_mrt    ← S1 weights, same input as training ✅
          └── Freq(x_base) → d_freq  ← S2' weights, same input as training ✅
        x = x_base + gate_mrt·d_mrt + gate_freq·d_freq → MLP

    Why parallel:
      - MRT was trained on raw residual (Stage 1), not Freq output
      - Freq was trained on raw residual (Stage 2'), not MRT output
      - Serial would give Freq MRT-enhanced input → domain shift
      - Parallel keeps each module's input consistent with its training
      - Gate (init balanced at 0.5/0.5) learns to blend the two experts

    Weight loading:
      - cycleQueue, mrt_blocks ← Stage 1 (MRT expert)
      - freq_blocks            ← Stage 2' (Freq expert)
      - gate_mrt/gate_freq     ← balanced init (0.5/0.5)
      - model (MLP)            ← Stage 1 (trained with MRT features)
    """
    print('\n' + '='*60)
    print("STAGE 3P: Merge MRT(S1) + Freq(S2') experts, PARALLEL fusion + gate")
    print('='*60)

    args.mrt_layers = 1
    args.freq_layers = 1
    args.fusion_mode = 'parallel'
    args.fusion_gate = 1
    args.freq_loss_alpha = 1.0  # pure MSE
    args.train_epochs = 30
    args.patience = 15

    setting = make_setting(args, '3p')
    ckpt_dir = os.path.join(args.checkpoints, setting)
    os.makedirs(ckpt_dir, exist_ok=True)

    train_loader, val_loader, test_loader = get_data_loaders(args)

    model = Model(args).float().to(device)

    # --- Override gate init to balanced (both experts are trustworthy) ---
    # Default init: MRT=2.0 (≈0.88), Freq=-1.5 (≈0.18) — biased toward MRT
    # Plan C: both are trained experts → balanced init at sigmoid(0)=0.5
    model.gate_mrt_logit.data.fill_(0.0)
    model.gate_freq_logit.data.fill_(0.0)
    print('[PlanC gate] init gate_mrt ≈ 0.5000, gate_freq ≈ 0.5000 (balanced experts)')

    # --- Selective weight loading ---
    s1_state = torch.load(stage1_ckpt)
    s2p_state = torch.load(stage2p_ckpt)
    model_state = model.state_dict()

    loaded_from_s1, loaded_from_s2p, fresh = 0, 0, 0
    for k in model_state.keys():
        # Gate params ← fresh (balanced init, don't overwrite)
        if k.startswith('gate_mrt_logit') or k.startswith('gate_freq_logit'):
            fresh += 1
        # Freq blocks ← Stage 2' (Freq expert, trained on raw residual)
        elif k.startswith('freq_blocks') or k.startswith('freq_v') or k.startswith('sgf_blocks'):
            if k in s2p_state and s2p_state[k].shape == model_state[k].shape:
                model_state[k] = s2p_state[k]
                loaded_from_s2p += 1
            else:
                fresh += 1
        # MRT + cycle + MLP ← Stage 1 (MRT expert, trained on raw residual)
        elif k in s1_state and s1_state[k].shape == model_state[k].shape:
            model_state[k] = s1_state[k]
            loaded_from_s1 += 1
        else:
            fresh += 1

    model.load_state_dict(model_state)
    print('Loaded: {} from S1(MRT), {} from S2\'(Freq), {} fresh'.format(
        loaded_from_s1, loaded_from_s2p, fresh))

    # Freeze nothing — full fine-tune
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Trainable: {:,} params (all unfrozen)'.format(n_trainable))

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate * 0.3)
    scheduler = lr_scheduler.OneCycleLR(
        optimizer, steps_per_epoch=len(train_loader),
        pct_start=args.pct_start, epochs=args.train_epochs, max_lr=args.learning_rate * 0.3)

    early_stopping = EarlyStopping(patience=args.patience, verbose=True)

    for epoch in range(args.train_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, args, scheduler, epoch)
        val_loss = validate(model, val_loader, criterion, device, args)
        test_loss = validate(model, test_loader, criterion, device, args)
        print('Epoch: {}, Steps: {} | Train: {:.7f} Vali: {:.7f} Test: {:.7f}'.format(
            epoch + 1, len(train_loader), train_loss, val_loss, test_loss))
        _diagnose_module_state(model, '3p-{}'.format(epoch + 1))

        early_stopping(val_loss, model, ckpt_dir)
        if early_stopping.early_stop:
            print('Early stopping at epoch {}'.format(epoch + 1))
            break

        if args.lradj != 'TST':
            adjust_learning_rate(optimizer, scheduler, epoch + 1, args)

    best_path = os.path.join(ckpt_dir, 'checkpoint.pth')
    model.load_state_dict(torch.load(best_path))
    mse, mae, smape = test_model(model, test_loader, device, args, setting)

    print('\n' + '='*60)
    print('FINAL RESULT (Stage 3P): mse={:.4f}, mae={:.4f}, smape={:.4f}'.format(mse, mae, smape))
    print('='*60)
    _diagnose_module_state(model, '3p-final')

    with open('result.txt', 'a') as f:
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, smape:{}\n\n'.format(mse, mae, smape))

    return mse, mae, smape


def stage2_add_freq_and_freeze(args, device, stage1_ckpt, cli_args=None):
    """Load MRT checkpoint, add Freq branch, freeze backbone+MRT+cycle, train Freq only."""
    max_epochs = getattr(cli_args, 'stage2_epochs', 3) if cli_args else 3
    patience = getattr(cli_args, 'stage2_patience', 3) if cli_args else 3

    print('\n' + '='*60)
    print('STAGE 2: Load MRT + add Freq branch (frozen backbone/MRT/cycle, max_epochs={}, patience={})'.format(
        max_epochs, patience))
    print('='*60)

    args.mrt_layers = 1
    args.freq_layers = 1
    args.freq_loss_alpha = 1.0  # pure MSE for warmup
    args.train_epochs = max_epochs
    args.patience = patience

    setting = make_setting(args, 2)
    ckpt_dir = os.path.join(args.checkpoints, setting)
    os.makedirs(ckpt_dir, exist_ok=True)

    train_loader, val_loader, test_loader = get_data_loaders(args)

    model = Model(args).float().to(device)

    stage1_state = torch.load(stage1_ckpt)
    model_state = model.state_dict()

    matched, unmatched = 0, 0
    for k, v in stage1_state.items():
        if k in model_state and v.shape == model_state[k].shape:
            model_state[k] = v
            matched += 1
        else:
            unmatched += 1
    model.load_state_dict(model_state)
    print('Loaded {} params from stage1, {} new (Freq branch)'.format(matched, unmatched))

    freeze_backbone = getattr(cli_args, 'stage2_freeze_backbone', 1) if cli_args else 1
    frozen_prefixes = ['cycleQueue', 'mrt_blocks']
    if freeze_backbone:
        frozen_prefixes.append('model')

    for name, param in model.named_parameters():
        should_freeze = any(name.startswith(p) for p in frozen_prefixes)
        param.requires_grad = not should_freeze

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print('Trainable: {:,} / {:,} ({:.1f}%)  freeze_backbone={}'.format(
        n_trainable, n_total, 100*n_trainable/n_total, freeze_backbone))

    criterion = nn.MSELoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    scheduler = lr_scheduler.OneCycleLR(
        optimizer, steps_per_epoch=len(train_loader),
        pct_start=args.pct_start, epochs=args.train_epochs, max_lr=args.learning_rate)

    early_stopping = EarlyStopping(patience=args.patience, verbose=True)

    for epoch in range(args.train_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, args, scheduler, epoch)
        val_loss = validate(model, val_loader, criterion, device, args)
        print('Epoch: {}, Train: {:.7f} Vali: {:.7f}'.format(epoch + 1, train_loss, val_loss))
        _diagnose_module_state(model, 's2-{}'.format(epoch + 1))

        early_stopping(val_loss, model, ckpt_dir)
        if early_stopping.early_stop:
            break

    stage2_path = os.path.join(ckpt_dir, 'checkpoint.pth')
    if not os.path.exists(stage2_path):
        torch.save(model.state_dict(), stage2_path)
    else:
        model.load_state_dict(torch.load(stage2_path))

    print('STAGE 2 complete.')
    return stage2_path


def stage3a_align_freq_model(args, device, stage2_ckpt, cli_args=None):
    """Progressive unfreeze part 1: freeze cycleQueue + mrt_blocks, train model + freq_blocks.

    Goal: let MLP backbone learn to consume Freq output before MRT competes for gradients.
    """
    max_epochs = getattr(cli_args, 'stage3a_epochs', 12) if cli_args else 12
    patience = getattr(cli_args, 'stage3a_patience', 6) if cli_args else 6

    print('\n' + '='*60)
    print('STAGE 3a: Freeze cycle+MRT, train model+Freq (max_epochs={}, patience={})'.format(
        max_epochs, patience))
    print('='*60)

    args.mrt_layers = 1
    args.freq_layers = 1
    args.freq_loss_alpha = 1.0  # pure MSE for alignment
    args.train_epochs = max_epochs
    args.patience = patience

    setting = make_setting(args, '3a')
    ckpt_dir = os.path.join(args.checkpoints, setting)
    os.makedirs(ckpt_dir, exist_ok=True)

    train_loader, val_loader, test_loader = get_data_loaders(args)

    model = Model(args).float().to(device)

    stage2_state = torch.load(stage2_ckpt)
    model_state = model.state_dict()
    matched, unmatched = 0, 0
    for k, v in stage2_state.items():
        if k in model_state and v.shape == model_state[k].shape:
            model_state[k] = v
            matched += 1
        else:
            unmatched += 1
    model.load_state_dict(model_state)
    print('Loaded {} params from stage2, {} new/mismatched'.format(matched, unmatched))

    # Freeze cycleQueue + mrt_blocks only; model + freq_blocks remain trainable
    frozen_prefixes = ['cycleQueue', 'mrt_blocks']
    for name, param in model.named_parameters():
        if any(name.startswith(p) for p in frozen_prefixes):
            param.requires_grad = False

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print('Trainable: {:,} / {:,} ({:.1f}%)'.format(n_trainable, n_total, 100*n_trainable/n_total))
    print('Frozen prefixes: {}'.format(frozen_prefixes))

    criterion = nn.MSELoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    scheduler = lr_scheduler.OneCycleLR(
        optimizer, steps_per_epoch=len(train_loader),
        pct_start=args.pct_start, epochs=max_epochs, max_lr=args.learning_rate)

    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    best_epoch = 0

    for epoch in range(max_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, args, scheduler, epoch)
        val_loss = validate(model, val_loader, criterion, device, args)
        test_loss = validate(model, test_loader, criterion, device, args)
        print('Epoch: {}, Steps: {} | Train: {:.7f} Vali: {:.7f} Test: {:.7f}'.format(
            epoch + 1, len(train_loader), train_loss, val_loss, test_loss))
        _diagnose_module_state(model, '3a-{}'.format(epoch + 1))

        early_stopping(val_loss, model, ckpt_dir)
        if early_stopping.early_stop:
            print('Early stopping at epoch {}'.format(epoch + 1))
            best_epoch = epoch + 1
            break

        if args.lradj != 'TST':
            adjust_learning_rate(optimizer, scheduler, epoch + 1, args)

    if best_epoch == 0:
        best_epoch = max_epochs

    stage3a_path = os.path.join(ckpt_dir, 'checkpoint.pth')
    if not os.path.exists(stage3a_path):
        torch.save(model.state_dict(), stage3a_path)
    else:
        model.load_state_dict(torch.load(stage3a_path))

    # Final diagnostics
    print('\n--- Stage 3a final diagnostics ---')
    _diagnose_module_state(model, '3a-final')

    # Acceptance checks (warnings only, don't block)
    if hasattr(model, 'freq_blocks') and len(model.freq_blocks) > 0:
        rs = model.freq_blocks[0].res_scale.item()
        mask = torch.sigmoid(model.freq_blocks[0].freq_weights)
        if abs(rs) < 0.01:
            print('  WARNING: freq.res_scale ≈ 0 — Freq branch did not activate.')
        if mask.std().item() < 0.01:
            print('  WARNING: freq mask std ≈ 0 — no frequency selectivity learned.')
    if hasattr(model, 'gate_freq_logit'):
        gf = torch.sigmoid(model.gate_freq_logit).mean().item()
        if gf < 0.03:
            print('  WARNING: gate_freq={:.4f} still near 0 — Freq branch suppressed.'.format(gf))

    print('STAGE 3a complete.')
    return stage3a_path


def stage3b_full_finetune(args, device, stage3a_ckpt, use_fredf=False, cli_args=None):
    """Progressive unfreeze part 2: unfreeze all, group-wise LR fine-tune.

    Group LR (multipliers on base_lr):
      - freq_blocks / gate:  high LR to keep learning frequency structure
      - model (MLP):         medium LR to refine forecasting
      - cycleQueue / mrt:    low LR to preserve what was learned in Stage 1-2
    """
    max_epochs = getattr(cli_args, 'stage3b_epochs', 20) if cli_args else 20
    patience = getattr(cli_args, 'stage3b_patience', 10) if cli_args else 10
    freq_mult = getattr(cli_args, 'stage3b_freq_lr_mult', 0.67) if cli_args else 0.67
    model_mult = getattr(cli_args, 'stage3b_model_lr_mult', 0.33) if cli_args else 0.33
    mrt_mult = getattr(cli_args, 'stage3b_mrt_lr_mult', 0.10) if cli_args else 0.10

    base_lr = args.learning_rate * 0.3  # reduce from dataset base LR
    fredf_tag = ' + FreDF(α=0.8)' if use_fredf else ' (pure MSE)'

    print('\n' + '='*60)
    print('STAGE 3b: Unfreeze all, group-wise LR (base={:.6f}, max_epochs={}, patience={}){}'.format(
        base_lr, max_epochs, patience, fredf_tag))
    print('  LR groups: freq/gate ×{:.2f}, model ×{:.2f}, mrt/cycle ×{:.2f}'.format(
        freq_mult, model_mult, mrt_mult))
    print('='*60)

    args.mrt_layers = 1
    args.freq_layers = 1
    args.freq_loss_alpha = 0.8 if use_fredf else 1.0
    args.train_epochs = max_epochs
    args.patience = patience

    setting = make_setting(args, '3b')
    ckpt_dir = os.path.join(args.checkpoints, setting)
    os.makedirs(ckpt_dir, exist_ok=True)

    train_loader, val_loader, test_loader = get_data_loaders(args)

    model = Model(args).float().to(device)
    model.load_state_dict(torch.load(stage3a_ckpt))
    print('Loaded stage3a checkpoint, unfreezing all parameters.')

    # Unfreeze all
    for param in model.parameters():
        param.requires_grad = True

    # Build group-wise parameter groups
    freq_prefixes = ['freq_blocks', 'freq_v2_blocks', 'freq_v3_blocks', 'freq_v4_blocks', 'sgf_blocks',
                     'gate_mrt_logit', 'gate_freq_logit']
    mrt_prefixes = ['cycleQueue', 'mrt_blocks']

    freq_params, model_params, mrt_params, other_params = [], [], [], []
    for name, param in model.named_parameters():
        if any(name.startswith(p) for p in freq_prefixes):
            freq_params.append(param)
        elif name.startswith('model.'):
            model_params.append(param)
        elif any(name.startswith(p) for p in mrt_prefixes):
            mrt_params.append(param)
        else:
            other_params.append(param)

    param_groups = []
    if freq_params:
        param_groups.append({'params': freq_params, 'lr': base_lr * freq_mult, 'name': 'freq/gate'})
    if model_params:
        param_groups.append({'params': model_params, 'lr': base_lr * model_mult, 'name': 'model'})
    if mrt_params:
        param_groups.append({'params': mrt_params, 'lr': base_lr * mrt_mult, 'name': 'mrt/cycle'})
    if other_params:
        param_groups.append({'params': other_params, 'lr': base_lr * 0.1, 'name': 'other'})

    for pg in param_groups:
        n = sum(p.numel() for p in pg['params'])
        print('  Group {}: {:,} params, LR={:.6f}'.format(pg['name'], n, pg['lr']))

    criterion = nn.MSELoss()
    optimizer = optim.Adam(param_groups, lr=base_lr)
    scheduler = lr_scheduler.OneCycleLR(
        optimizer, steps_per_epoch=len(train_loader),
        pct_start=args.pct_start, epochs=max_epochs, max_lr=base_lr)

    early_stopping = EarlyStopping(patience=args.patience, verbose=True)

    for epoch in range(max_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, args, scheduler, epoch)
        val_loss = validate(model, val_loader, criterion, device, args)
        test_loss = validate(model, test_loader, criterion, device, args)
        print('Epoch: {}, Steps: {} | Train: {:.7f} Vali: {:.7f} Test: {:.7f}'.format(
            epoch + 1, len(train_loader), train_loss, val_loss, test_loss))
        _diagnose_module_state(model, '3b-{}'.format(epoch + 1))

        early_stopping(val_loss, model, ckpt_dir)
        if early_stopping.early_stop:
            print('Early stopping at epoch {}'.format(epoch + 1))
            break

        if args.lradj != 'TST':
            adjust_learning_rate(optimizer, scheduler, epoch + 1, args)

    best_path = os.path.join(ckpt_dir, 'checkpoint.pth')
    model.load_state_dict(torch.load(best_path))
    mse, mae, smape = test_model(model, test_loader, device, args, setting)

    print('\n' + '='*60)
    print('FINAL RESULT (Stage 3b): mse={:.4f}, mae={:.4f}, smape={:.4f}'.format(mse, mae, smape))
    print('='*60)
    _diagnose_module_state(model, '3b-final')

    with open('result.txt', 'a') as f:
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, smape:{}\n\n'.format(mse, mae, smape))

    return mse, mae, smape


# ─── Main ──────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['etth1', 'etth2', 'ettm1', 'ettm2', 'weather', 'electricity', 'traffic', 'solar'])
    parser.add_argument('--pred_len', type=int, required=True, choices=[96, 192, 336, 720])
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--gpu', type=int, default=0, help='GPU device id')
    parser.add_argument('--stage1_only', action='store_true', help='Only run stage 1')
    parser.add_argument('--stage2_only', action='store_true', help='Stop after Stage 2, test and exit')
    parser.add_argument('--stage2_freeze_backbone', type=int, default=1,
                        help='1=freeze MLP in Stage 2, 0=unfreeze (train Freq+MLP jointly)')
    parser.add_argument('--stage3_fredf', action='store_true', help='Enable FreDF (α=0.8) in Stage 3b')
    parser.add_argument('--from_stage2', type=str, default='', help='Skip to stage 2 with given stage1 ckpt path')
    parser.add_argument('--plan_c', action='store_true',
                        help='Plan C: train MRT & Freq as separate experts, then merge')
    # Model architecture
    parser.add_argument('--fusion_mode', type=str, default='serial', choices=['serial', 'parallel'],
                        help='Enhancement fusion: serial (default) or parallel with residual-delta fusion')
    parser.add_argument('--fusion_gate', type=int, default=0,
                        help='Parallel fusion gate: 0=plain sum, 1=per-channel sigmoid gate')
    # Stage 2 (Freq warmup: frozen backbone+MRT+cycle, train Freq only)
    parser.add_argument('--stage2_epochs', type=int, default=3,
                        help='Stage 2 max epochs (freq warmup, frozen backbone/MRT/cycle)')
    parser.add_argument('--stage2_patience', type=int, default=3,
                        help='Stage 2 early stopping patience')
    # Stage 3a (progressive unfreeze: freeze cycle+mrt, train model+freq)
    parser.add_argument('--stage3a_epochs', type=int, default=12,
                        help='Stage 3a max epochs (frozen cycle+mrt, train model+freq)')
    parser.add_argument('--stage3a_patience', type=int, default=6,
                        help='Stage 3a early stopping patience')
    # Stage 3b (full unfreeze + group-wise LR)
    parser.add_argument('--stage3b_epochs', type=int, default=20,
                        help='Stage 3b max epochs (group-wise LR full finetune)')
    parser.add_argument('--stage3b_patience', type=int, default=10,
                        help='Stage 3b early stopping patience')
    parser.add_argument('--stage3b_freq_lr_mult', type=float, default=0.67,
                        help='LR multiplier for freq/gate params in stage 3b')
    parser.add_argument('--stage3b_model_lr_mult', type=float, default=0.33,
                        help='LR multiplier for MLP backbone in stage 3b')
    parser.add_argument('--stage3b_mrt_lr_mult', type=float, default=0.10,
                        help='LR multiplier for mrt/cycle params in stage 3b')
    parser.add_argument('--freq_sparsity_lambda', type=float, default=0.0,
                        help='Sparsity regularizer weight: pushes freq mask away from 0.5 (0=off, 0.01~0.1 recommended)')
    parser.add_argument('--no_stage3a', action='store_true',
                        help='Skip Stage 3a, go directly Stage 2 -> Stage 3b (avoids freeze-then-unfreeze distribution shift)')
    args_cli = parser.parse_args()

    os.chdir(ROOT)
    device = torch.device('cuda:{}'.format(args_cli.gpu) if torch.cuda.is_available() else 'cpu')
    dataset_name = args_cli.dataset
    base_args = build_args(dataset_name, args_cli.pred_len, args_cli.seed)

    # Override with CLI fusion settings
    base_args.fusion_mode = args_cli.fusion_mode
    base_args.fusion_gate = args_cli.fusion_gate
    base_args.freq_sparsity_lambda = args_cli.freq_sparsity_lambda

    print('='*60)
    print('CONFIGURATION')
    print('='*60)
    print('Device     :', device)
    print('Dataset    :', dataset_name)
    print('Pred len   :', args_cli.pred_len)
    print('Seed       :', args_cli.seed)
    print('')
    print('--- Data ---')
    print('enc_in     :', base_args.enc_in)
    print('seq_len    :', base_args.seq_len)
    print('cycle      :', base_args.cycle)
    print('use_revin  :', base_args.use_revin)
    print('batch_size :', base_args.batch_size)
    print('')
    print('--- Model ---')
    print('model_type :', base_args.model_type)
    print('d_model    :', base_args.d_model)
    print('mrt_layers :', base_args.mrt_layers)
    print('freq_layers:', base_args.freq_layers)
    print('fusion     : mode={}, gate={}, sparsity={}'.format(
        base_args.fusion_mode, base_args.fusion_gate, base_args.freq_sparsity_lambda))
    print('')
    print('--- Training ---')
    print('base_lr    :', base_args.learning_rate)
    print('Stage1     : epochs={}, patience={}'.format(base_args.train_epochs, base_args.patience))
    freeze_bb = getattr(args_cli, 'stage2_freeze_backbone', 1)
    print('Stage2     : epochs={}, patience={}, freeze_backbone={}'.format(
        args_cli.stage2_epochs, args_cli.stage2_patience, freeze_bb))
    if args_cli.stage2_only:
        print('Stage3     : SKIPPED (stage2_only)')
    else:
        print('Stage3 FreDF:', args_cli.stage3_fredf)
        print('Stage3a    : epochs={}, patience={}'.format(args_cli.stage3a_epochs, args_cli.stage3a_patience))
        print('Stage3b    : epochs={}, patience={}, freq×{:.2f} model×{:.2f} mrt×{:.2f}'.format(
            args_cli.stage3b_epochs, args_cli.stage3b_patience,
            args_cli.stage3b_freq_lr_mult, args_cli.stage3b_model_lr_mult, args_cli.stage3b_mrt_lr_mult))
    print('='*60)

    # --- Stage 1: Train MRT only ---
    if args_cli.from_stage2:
        stage1_path = args_cli.from_stage2
        print('Skipping stage 1, using checkpoint:', stage1_path)
    else:
        stage1_path = stage1_train_mrt(base_args, device)

    if args_cli.stage1_only:
        print('Stage 1 only. Done.')
        sys.exit(0)

    # ── Plan C: separate experts → merge ──
    if args_cli.plan_c:
        print('\n' + '#'*60)
        print('#  PLAN C: MRT & Freq as separate experts → merge')
        print('#'*60)

        # Stage 2': Train Freq in MRT context (parallel+gate, MRT+MLP frozen)
        stage2p_path = stage2p_train_freq_only(base_args, device, stage1_path)

        # Stage 3P: Merge MRT & Freq ckpts, joint fine-tune
        stage3p_merge_and_finetune(base_args, device, stage1_path, stage2p_path)
        print('Plan C complete.')
        sys.exit(0)

    # ── Original flow: Stage 2 → Stage 3a → Stage 3b ──
    stage2_path = stage2_add_freq_and_freeze(base_args, device, stage1_path, cli_args=args_cli)

    if args_cli.stage2_only:
        # Test Stage 2 model and exit
        print('\n' + '='*60)
        print('STAGE 2 ONLY: Testing from Stage 2 checkpoint')
        print('='*60)
        model = Model(base_args).float().to(device)
        model.load_state_dict(torch.load(stage2_path))
        model.eval()
        _, _, test_loader = get_data_loaders(base_args)
        mse, mae, smape = test_model(model, test_loader, device, base_args, base_args.model_id + '_stage2')
        _diagnose_module_state(model, 'stage2-final')
        print('\n' + '='*60)
        print('FINAL RESULT (Stage 2 only): mse={:.4f}, mae={:.4f}, smape={:.4f}'.format(mse, mae, smape))
        print('='*60)
        sys.exit(0)

    if args_cli.no_stage3a:
        # Skip Stage 3a: avoids freeze-then-unfreeze distribution shift.
        # Go directly Stage 2 → Stage 3b with group-wise LR.
        print('\n' + '='*60)
        print('SKIPPING Stage 3a (--no_stage3a): Stage 2 -> Stage 3b directly')
        print('='*60)
        stage3b_full_finetune(base_args, device, stage2_path, use_fredf=args_cli.stage3_fredf, cli_args=args_cli)
    else:
        # --- Stage 3a: Freeze cycle+mrt, train model+freq (progressive unfreeze) ---
        stage3a_path = stage3a_align_freq_model(base_args, device, stage2_path, cli_args=args_cli)

        # --- Stage 3b: Unfreeze all, group-wise LR fine-tune ---
        stage3b_full_finetune(base_args, device, stage3a_path, use_fredf=args_cli.stage3_fredf, cli_args=args_cli)
