#!/usr/bin/env python
"""
2-Stage training v2: MRT warmstart + MRT+Freq joint finetune with diff LR.

Stage 1: Train MRT only, lr=default, save checkpoint.
Stage 2: Load MRT, unfreeze all, joint fine-tune with:
         - freq_blocks: lr=0.005
         - all others:  lr=0.001
         Freq res_scale init = -0.1 (subtractive filter bias)

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/multi_stage_train_v2.py \
        --dataset etth1 --pred_len 720 [--gpu 0]
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

ROOT = '/data/data_huaji/timeSeries/CycleNetBaseLine'

DATASET_CONFIGS = {
    'etth1': {
        'data': 'ETTh1', 'root_path': './dataset/ETT-small/', 'data_path': 'ETTh1.csv',
        'enc_in': 7, 'cycle': 24, 'use_revin': 1, 'batch_size': 256, 'lr': 0.005,
        'freq': 'h', 'stage2_epochs': 25, 'stage2_patience': 12,
    },
    'etth2': {
        'data': 'ETTh2', 'root_path': './dataset/ETT-small/', 'data_path': 'ETTh2.csv',
        'enc_in': 7, 'cycle': 24, 'use_revin': 1, 'batch_size': 256, 'lr': 0.005,
        'freq': 'h', 'stage2_epochs': 25, 'stage2_patience': 12,
    },
    'ettm1': {
        'data': 'ETTm1', 'root_path': './dataset/ETT-small/', 'data_path': 'ETTm1.csv',
        'enc_in': 7, 'cycle': 96, 'use_revin': 1, 'batch_size': 256, 'lr': 0.005,
        'freq': 'h', 'stage2_epochs': 25, 'stage2_patience': 12,
    },
    'ettm2': {
        'data': 'ETTm2', 'root_path': './dataset/ETT-small/', 'data_path': 'ETTm2.csv',
        'enc_in': 7, 'cycle': 96, 'use_revin': 1, 'batch_size': 256, 'lr': 0.005,
        'freq': 'h', 'stage2_epochs': 25, 'stage2_patience': 12,
    },
    'solar': {
        'data': 'Solar', 'root_path': './dataset/Solar/', 'data_path': 'solar_AL.txt',
        'enc_in': 137, 'cycle': 144, 'use_revin': 0, 'batch_size': 64, 'lr': 0.01,
        'freq': 'h', 'stage2_epochs': 25, 'stage2_patience': 12,
    },
}


def build_args(dataset, pred_len, seed=2024):
    cfg = DATASET_CONFIGS[dataset]
    parser = argparse.ArgumentParser()
    parser.add_argument('--is_training', type=int, default=1)
    parser.add_argument('--model_id', type=str, default='{}_96_{}'.format(dataset.upper(), pred_len))
    parser.add_argument('--model', type=str, default='CycleNet')
    parser.add_argument('--data', type=str, default=cfg['data'])
    parser.add_argument('--root_path', type=str, default=cfg['root_path'])
    parser.add_argument('--data_path', type=str, default=cfg['data_path'])
    parser.add_argument('--features', type=str, default='M')
    parser.add_argument('--target', type=str, default='OT')
    parser.add_argument('--freq', type=str, default=cfg['freq'])
    parser.add_argument('--checkpoints', type=str, default='./checkpoints_new/')
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--label_len', type=int, default=0)
    parser.add_argument('--pred_len', type=int, default=pred_len)
    parser.add_argument('--cycle', type=int, default=cfg['cycle'])
    parser.add_argument('--cycle_mode', type=str, default='lookup')
    parser.add_argument('--cycle_rank', type=int, default=4)
    parser.add_argument('--model_type', type=str, default='mlp')
    parser.add_argument('--use_revin', type=int, default=cfg['use_revin'])
    parser.add_argument('--mrt_layers', type=int, default=1)
    parser.add_argument('--freq_layers', type=int, default=0)
    parser.add_argument('--freq_v2_layers', type=int, default=0)
    parser.add_argument('--freq_v3_layers', type=int, default=0)
    parser.add_argument('--freq_v4_layers', type=int, default=0)
    parser.add_argument('--sgf_layers', type=int, default=0)
    parser.add_argument('--sgf_prior_path', type=str, default='')
    parser.add_argument('--freq_loss_alpha', type=float, default=1.0)
    parser.add_argument('--freq_res_scale_init', type=float, default=0.0)
    parser.add_argument('--fusion_mode', type=str, default='serial')
    parser.add_argument('--fusion_order', type=str, default='mrt_freq')
    parser.add_argument('--fusion_gate', type=int, default=0)
    parser.add_argument('--enc_in', type=int, default=cfg['enc_in'])
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--fc_dropout', type=float, default=0.05)
    parser.add_argument('--head_dropout', type=float, default=0.0)
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
    # Dummy
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
    args.dataset_key = dataset
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
    mrt_l, freq_l = args.mrt_layers, args.freq_layers
    suffix = ''
    if mrt_l: suffix += '_mrt{}'.format(mrt_l)
    if freq_l: suffix += '_freq{}'.format(freq_l)
    if args.freq_loss_alpha < 1.0:
        suffix += '_fredf{}'.format(str(args.freq_loss_alpha).replace('.', ''))
    suffix += '_rs{}_ep{}_stage{}'.format(
        str(args.freq_res_scale_init).replace('.', '').replace('-', 'n'),
        args.train_epochs, stage)
    return '{}_{}_{}_ft{}_sl{}_pl{}_cycle{}_{}{}_seed{}'.format(
        args.model_id, args.model, args.data, args.features,
        args.seq_len, args.pred_len, args.cycle, args.model_type, suffix, seed)


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
    smape_val = float(200 * np.mean(np.abs(preds - trues) / (np.abs(preds) + np.abs(trues) + 1e-8)))
    folder = './test_results_new/{}/'.format(setting)
    os.makedirs(folder, exist_ok=True)
    np.save(folder + 'metrics.npy', np.array([mae, mse, rmse, float(mape), float(mspe), float(np.mean(rse)), float(np.mean(corr)), smape_val]))
    np.save(folder + 'pred.npy', preds)
    np.save(folder + 'true.npy', trues)
    np.save(folder + 'residual.npy', preds - trues)
    return mse, mae, smape_val


# ─── Stage 1: MRT only ─────────────────────────────────────────

def stage1_train_mrt(args, device):
    print('\n' + '='*60)
    print('STAGE 1: Train MRT only (freq_layers=0)')
    print('='*60)
    args.mrt_layers = 1
    args.freq_layers = 0
    args.freq_loss_alpha = 1.0
    args.train_epochs = 30
    args.patience = 15
    setting = make_setting(args, 1)
    ckpt_dir = os.path.join(args.checkpoints, setting)
    os.makedirs(ckpt_dir, exist_ok=True)
    train_loader, val_loader, test_loader = get_data_loaders(args)
    model = Model(args).float().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = lr_scheduler.OneCycleLR(optimizer, steps_per_epoch=len(train_loader),
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
    return best_path


# ─── Stage 2: MRT+Freq joint finetune with diff LR ─────────────

def stage2_joint_finetune(args, device, stage1_ckpt):
    cfg = DATASET_CONFIGS[args.dataset_key]
    stage2_epochs = cfg.get('stage2_epochs', 25)
    stage2_patience = cfg.get('stage2_patience', 12)

    print('\n' + '='*60)
    print('STAGE 2: MRT+Freq joint finetune with diff LR')
    print('  Freq LR={}, Others LR=0.001, res_scale_init=-0.1'.format(args.learning_rate))
    print('  Epochs={}, Patience={}'.format(stage2_epochs, stage2_patience))
    print('='*60)

    args.mrt_layers = 1
    args.freq_layers = 1
    args.freq_res_scale_init = 0.0  # zero-init: safe identity start
    args.freq_loss_alpha = 1.0
    args.train_epochs = stage2_epochs
    args.patience = stage2_patience

    setting = make_setting(args, 2)
    ckpt_dir = os.path.join(args.checkpoints, setting)
    os.makedirs(ckpt_dir, exist_ok=True)

    train_loader, val_loader, test_loader = get_data_loaders(args)

    model = Model(args).float().to(device)

    # Load Stage 1 weights for matching keys
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
    new_params = sum(1 for k in model_state if k not in stage1_state)
    print('Loaded {} params from stage1, {} new (Freq branch), {} unmatched'.format(matched, new_params, unmatched))

    # Diff LR: freq_blocks get base_lr, others get 0.001
    freq_lr = args.learning_rate
    other_lr = 0.001
    freq_params = []
    other_params = []
    for name, param in model.named_parameters():
        if name.startswith('freq_blocks'):
            freq_params.append(param)
        else:
            other_params.append(param)

    optimizer = optim.Adam([
        {'params': freq_params, 'lr': freq_lr},
        {'params': other_params, 'lr': other_lr},
    ])
    print('Param groups: freq_blocks (lr={:.4f}, {} params), others (lr={:.4f}, {} params)'.format(
        freq_lr, sum(p.numel() for p in freq_params),
        other_lr, sum(p.numel() for p in other_params)))

    criterion = nn.MSELoss()
    scheduler = lr_scheduler.OneCycleLR(optimizer, steps_per_epoch=len(train_loader),
        pct_start=args.pct_start, epochs=args.train_epochs, max_lr=[freq_lr, other_lr])

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
    print('\n' + '='*60)
    print('FINAL RESULT: mse={:.4f}, mae={:.4f}, smape={:.4f}'.format(mse, mae, smape))
    print('='*60)

    # Log to result file
    with open('result_v2.txt', 'a') as f:
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, smape:{}\n\n'.format(mse, mae, smape))

    return mse, mae, smape


# ─── Main ──────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['etth1', 'etth2', 'ettm1', 'ettm2', 'solar'])
    parser.add_argument('--pred_len', type=int, required=True, choices=[96, 192, 336, 720])
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--gpu', type=int, default=0, help='GPU device id (for CUDA_VISIBLE_DEVICES)')
    args_cli = parser.parse_args()

    os.chdir(ROOT)
    device = torch.device('cuda:0')  # CUDA_VISIBLE_DEVICES already set
    dataset_name = args_cli.dataset
    print('Device:', device)
    print('Dataset:', dataset_name, '| Pred len:', args_cli.pred_len, '| Seed:', args_cli.seed)

    base_args = build_args(dataset_name, args_cli.pred_len, args_cli.seed)

    stage1_path = stage1_train_mrt(base_args, device)
    stage2_joint_finetune(base_args, device, stage1_path)
