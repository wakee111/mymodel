#!/usr/bin/env python
"""
3-Stage training for ETTh1 with trend_freq mode (pure MSE).

Stage 1: Train MRT only (serial mode, MRT=1, Freq=0), save checkpoint.
         MRT learns to extract low-frequency trend from residual.
Stage 2: Switch to trend_freq mode. Load MRT checkpoint. Add Freq branch.
         Freeze backbone+MRT+cycleQueue. Train Freq + alpha_t + alpha_h.
         Freq processes detrended residual (r - detach(trend_delta)).
Stage 3: Unfreeze all. Joint fine-tune with lower LR in trend_freq mode.

Usage:
    CUDA_VISIBLE_DEVICES=5 python scripts/multi_stage_train_etth1_tf.py --pred_len 192
"""

import os, sys, argparse, time, random
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.CycleNet import Model
from data_provider.data_factory import data_provider
from utils.tools import EarlyStopping, adjust_learning_rate

PYTHON = '/data/data_linsf/Anaconda3/envs/TimeSeries_huaji/bin/python'
ROOT = '/data/data_huaji/timeSeries/CycleNetBaseLine'


def build_args(pred_len, seed=2024):
    """Build args for ETTh1 dataset."""
    parser = argparse.ArgumentParser()
    # Required
    parser.add_argument('--is_training', type=int, default=1)
    parser.add_argument('--model_id', type=str, default='ETTh1_96_{}'.format(pred_len))
    parser.add_argument('--model', type=str, default='CycleNet')
    parser.add_argument('--data', type=str, default='ETTh1')
    # Data
    parser.add_argument('--root_path', type=str, default='./dataset/ETT-small/')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv')
    parser.add_argument('--features', type=str, default='M')
    parser.add_argument('--target', type=str, default='OT')
    parser.add_argument('--freq', type=str, default='h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/')
    # Task
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--label_len', type=int, default=0)
    parser.add_argument('--pred_len', type=int, default=pred_len)
    # CycleNet
    parser.add_argument('--cycle', type=int, default=24)
    parser.add_argument('--cycle_mode', type=str, default='lookup')
    parser.add_argument('--cycle_rank', type=int, default=4)
    parser.add_argument('--model_type', type=str, default='mlp')
    parser.add_argument('--use_revin', type=int, default=1)
    # Enhancement modules (set per stage)
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
    parser.add_argument('--detach_trend_for_freq', type=int, default=1)
    # Model
    parser.add_argument('--enc_in', type=int, default=7)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--fc_dropout', type=float, default=0.05)
    parser.add_argument('--head_dropout', type=float, default=0.0)
    # Optim
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--itr', type=int, default=1)
    parser.add_argument('--train_epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--learning_rate', type=float, default=0.005)
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
    parser.add_argument('--devices', type=str, default='0,1')

    args = parser.parse_args([])

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    return args


def get_data_loaders(args):
    train_set, train_loader = data_provider(args, 'train')
    val_set, val_loader = data_provider(args, 'val')
    test_set, test_loader = data_provider(args, 'test')
    return train_loader, val_loader, test_loader


def make_setting(args, stage, seed=2024):
    """Build setting string including trend_freq suffix."""
    mrt_l = args.mrt_layers
    freq_l = args.freq_layers
    suffix = ''
    if mrt_l: suffix += '_mrt{}'.format(mrt_l)
    if freq_l: suffix += '_freq{}'.format(freq_l)
    if args.fusion_mode == 'trend_freq':
        suffix += '_tf'
    if args.freq_loss_alpha < 1.0:
        suffix += '_fredf{}'.format(str(args.freq_loss_alpha).replace('.', ''))
    suffix += '_stage{}'.format(stage)
    return '{}_{}_{}_ft{}_sl{}_pl{}_cycle{}_{}{}_seed{}'.format(
        args.model_id, args.model, args.data, args.features,
        args.seq_len, args.pred_len, args.cycle, args.model_type, suffix, seed)


def train_one_epoch(model, train_loader, optimizer, criterion, device, args,
                    scheduler=None, epoch=0):
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
            loss_freq = (torch.fft.rfft(outputs, dim=1) -
                         torch.fft.rfft(batch_y, dim=1)).abs().mean()
            loss = args.freq_loss_alpha * loss + (1 - args.freq_loss_alpha) * loss_freq

        loss.backward()
        optimizer.step()

        if args.lradj == 'TST' and scheduler:
            adjust_learning_rate(optimizer, scheduler, epoch + 1, args, printout=False)
            scheduler.step()

        total_loss.append(loss.item())

        if (i + 1) % 100 == 0:
            speed = (time.time() - time_now) / iter_count
            left = speed * ((args.train_epochs - epoch) * train_steps - i)
            # trend_freq: log alpha values
            extra = ''
            if getattr(args, 'fusion_mode', 'serial') == 'trend_freq':
                extra = ' | at={:.4f} ah={:.4f}'.format(
                    model.alpha_t.item(), model.alpha_h.item())
            print('    iters: {}, epoch: {} | loss: {:.7f}{} | speed: {:.4f}s/iter; left: {:.0f}s'.format(
                i + 1, epoch + 1, loss.item(), extra, speed, left))
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
    """Full test with metric saving."""
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

    folder = './test_results/{}/'.format(setting)
    os.makedirs(folder, exist_ok=True)
    np.save(folder + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe, rse, corr, smape_val]))
    np.save(folder + 'pred.npy', preds)
    np.save(folder + 'true.npy', trues)
    np.save(folder + 'residual.npy', preds - trues)

    return mse, mae, smape_val


# ─── Stage helpers ────────────────────────────────────────────────

def log_params(args, pred_len, seed):
    """Print all experiment parameters to log."""
    print('\n' + '=' * 60)
    print('EXPERIMENT PARAMETERS')
    print('=' * 60)
    print('  Dataset:        ETTh1')
    print('  Model:          CycleNet (mlp)')
    print('  Input:          seq_len=96, pred_len={}'.format(pred_len))
    print('  enc_in:         7, cycle=24, use_revin=1')
    print('  d_model:        512')
    print('  batch_size:     256')
    print('  Seed:           {}'.format(seed))
    print('  GPU:            {}'.format(args.gpu))
    print('---')
    print('  Stage 1:  serial,   MRT=1, Freq=0, epochs=30, lr=0.005, patience=15')
    print('  Stage 2:  trend_freq, MRT=1, Freq=1, epochs=3,  lr=0.005, patience=3')
    print('            freeze=[backbone,MRT,cycle] train=[Freq,alpha_t,alpha_h]')
    print('            detach_trend_for_freq=1')
    print('  Stage 3:  trend_freq, MRT=1, Freq=1, epochs=30, lr=0.0005, patience=15')
    print('            unfreeze all, joint fine-tune')
    print('---')
    print('  Loss:           Pure MSE (freq_loss_alpha=1.0)')
    print('  alpha_t init:   0.0 (zero-init, MRT trend blend)')
    print('  alpha_h init:   0.1 (non-zero to avoid Freq deadlock)')
    print('=' * 60 + '\n')


def stage1_train_mrt(args, device):
    """Stage 1: Train MRT only in serial mode (pure MSE)."""
    print('\n' + '=' * 60)
    print('STAGE 1: Train MRT only (serial mode, mrt=1, freq=0)')
    print('=' * 60)

    args.mrt_layers = 1
    args.freq_layers = 0
    args.fusion_mode = 'serial'
    args.freq_loss_alpha = 1.0
    args.train_epochs = 30
    args.patience = 15
    args.learning_rate = 0.005

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
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion,
                                     device, args, scheduler, epoch)
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


def stage2_add_freq_in_tf_mode(args, device, stage1_ckpt):
    """Stage 2: Load MRT, switch to trend_freq mode, add Freq branch.
    Freeze backbone + MRT + cycleQueue. Train Freq + alpha_t + alpha_h."""
    print('\n' + '=' * 60)
    print('STAGE 2: trend_freq mode - Add Freq, freeze backbone/MRT/cycle')
    print('=' * 60)

    args.mrt_layers = 1
    args.freq_layers = 1
    args.fusion_mode = 'trend_freq'
    args.detach_trend_for_freq = 1
    args.freq_loss_alpha = 1.0
    args.train_epochs = 3
    args.patience = 3

    setting = make_setting(args, 2)
    ckpt_dir = os.path.join(args.checkpoints, setting)
    os.makedirs(ckpt_dir, exist_ok=True)

    train_loader, val_loader, test_loader = get_data_loaders(args)

    model = Model(args).float().to(device)

    # Load stage1 weights (keys that match)
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
    print('Loaded {} params from stage1, {} new (Freq+alpha_t+alpha_h)'.format(matched, unmatched))
    print('alpha_t={:.6f}, alpha_h={:.6f} (zero-init)'.format(
        model.alpha_t.item(), model.alpha_h.item()))

    # Freeze backbone, MRT, cycleQueue → only Freq + alpha_t + alpha_h trainable
    frozen_prefixes = ['cycleQueue', 'mrt_blocks', 'model']
    for name, param in model.named_parameters():
        should_freeze = any(name.startswith(p) for p in frozen_prefixes)
        param.requires_grad = not should_freeze

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print('Trainable: {:,} / {:,} ({:.1f}%)'.format(
        n_trainable, n_total, 100 * n_trainable / n_total))

    criterion = nn.MSELoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=args.learning_rate)
    scheduler = lr_scheduler.OneCycleLR(
        optimizer, steps_per_epoch=len(train_loader),
        pct_start=args.pct_start, epochs=args.train_epochs, max_lr=args.learning_rate)

    early_stopping = EarlyStopping(patience=args.patience, verbose=True)

    for epoch in range(args.train_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion,
                                     device, args, scheduler, epoch)
        val_loss = validate(model, val_loader, criterion, device, args)
        print('Epoch: {}, Train: {:.7f} Vali: {:.7f} | alpha_t={:.6f} alpha_h={:.6f}'.format(
            epoch + 1, train_loss, val_loss,
            model.alpha_t.item(), model.alpha_h.item()))

        early_stopping(val_loss, model, ckpt_dir)
        if early_stopping.early_stop:
            break

    stage2_path = os.path.join(ckpt_dir, 'checkpoint.pth')
    if not os.path.exists(stage2_path):
        torch.save(model.state_dict(), stage2_path)
    else:
        model.load_state_dict(torch.load(stage2_path))
    print('STAGE 2 complete. alpha_t={:.6f}, alpha_h={:.6f}'.format(
        model.alpha_t.item(), model.alpha_h.item()))
    return stage2_path


def stage3_joint_finetune(args, device, stage2_ckpt):
    """Stage 3: Unfreeze all, joint fine-tune in trend_freq mode (low LR)."""
    print('\n' + '=' * 60)
    print('STAGE 3: trend_freq mode - Unfreeze all, joint fine-tune (LR=0.0005)')
    print('=' * 60)

    args.mrt_layers = 1
    args.freq_layers = 1
    args.fusion_mode = 'trend_freq'
    args.detach_trend_for_freq = 1
    args.freq_loss_alpha = 1.0
    args.train_epochs = 30
    args.patience = 15
    args.learning_rate = 0.0005  # lower LR for joint tuning

    setting = make_setting(args, 3)
    ckpt_dir = os.path.join(args.checkpoints, setting)
    os.makedirs(ckpt_dir, exist_ok=True)

    train_loader, val_loader, test_loader = get_data_loaders(args)

    model = Model(args).float().to(device)
    model.load_state_dict(torch.load(stage2_ckpt))
    print('Loaded stage2 checkpoint. alpha_t={:.6f}, alpha_h={:.6f}'.format(
        model.alpha_t.item(), model.alpha_h.item()))

    # Unfreeze all
    for param in model.parameters():
        param.requires_grad = True

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = lr_scheduler.OneCycleLR(
        optimizer, steps_per_epoch=len(train_loader),
        pct_start=args.pct_start, epochs=args.train_epochs, max_lr=args.learning_rate)

    early_stopping = EarlyStopping(patience=args.patience, verbose=True)

    for epoch in range(args.train_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion,
                                     device, args, scheduler, epoch)
        val_loss = validate(model, val_loader, criterion, device, args)
        test_loss = validate(model, test_loader, criterion, device, args)
        print('Epoch: {}, Steps: {} | Train: {:.7f} Vali: {:.7f} Test: {:.7f} | '
              'alpha_t={:.6f} alpha_h={:.6f}'.format(
            epoch + 1, len(train_loader), train_loss, val_loss, test_loss,
            model.alpha_t.item(), model.alpha_h.item()))

        early_stopping(val_loss, model, ckpt_dir)
        if early_stopping.early_stop:
            print('Early stopping at epoch {}'.format(epoch + 1))
            break

        if args.lradj != 'TST':
            adjust_learning_rate(optimizer, scheduler, epoch + 1, args)

    # Load best and test
    best_path = os.path.join(ckpt_dir, 'checkpoint.pth')
    model.load_state_dict(torch.load(best_path))
    mse, mae, smape = test_model(model, test_loader, device, args, setting)
    print('\n' + '=' * 60)
    print('FINAL RESULT: mse={:.4f}, mae={:.4f}, smape={:.4f}'.format(mse, mae, smape))
    print('alpha_t={:.6f}, alpha_h={:.6f}'.format(
        model.alpha_t.item(), model.alpha_h.item()))
    print('=' * 60)

    with open('result.txt', 'a') as f:
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, smape:{}\n'.format(mse, mae, smape))
        f.write('alpha_t:{}, alpha_h:{}\n\n'.format(
            model.alpha_t.item(), model.alpha_h.item()))

    return mse, mae, smape


# ─── Main ──────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_len', type=int, required=True,
                        choices=[96, 192, 336, 720])
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--gpu', type=int, default=5)
    parser.add_argument('--stage1_only', action='store_true')
    parser.add_argument('--from_stage2', type=str, default='')
    args_cli = parser.parse_args()

    os.chdir(ROOT)
    device = torch.device('cuda:{}'.format(args_cli.gpu)
                          if torch.cuda.is_available() else 'cpu')
    print('Device:', device)
    print('Pred len:', args_cli.pred_len, 'Seed:', args_cli.seed)

    base_args = build_args(args_cli.pred_len, args_cli.seed)
    log_params(base_args, args_cli.pred_len, args_cli.seed)

    if args_cli.from_stage2:
        stage1_path = args_cli.from_stage2
        print('Skipping stage 1, using checkpoint:', stage1_path)
    else:
        stage1_path = stage1_train_mrt(base_args, device)

    if args_cli.stage1_only:
        print('Stage 1 only. Done.')
        sys.exit(0)

    stage2_path = stage2_add_freq_in_tf_mode(base_args, device, stage1_path)
    stage3_joint_finetune(base_args, device, stage2_path)
