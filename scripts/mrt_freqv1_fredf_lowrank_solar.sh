#!/bin/bash
# Solar MRT + FreqV1 + FreDF + LowRankCycle
# seq=96, cycle=144 (daily), enc_in=137, 10-min data, use_revin=0
# seed=2024, patience=15

cd /data/data_huaji/timeSeries/CycleNetBaseLine

log_dir=./logs/mrt_freqv1_fredf_lowrank_solar
mkdir -p $log_dir

for pred_len in 96 192 336 720; do
    echo "=== pred_len=$pred_len ==="

    echo "  MRT=1 + FreqV1=1 + FreDF(α=0.8) + LowRankCycle(rank=4)"
    CUDA_VISIBLE_DEVICES=1 python -u run.py \
      --is_training 1 --root_path ./dataset/Solar/ --data_path solar_AL.txt \
      --model_id Solar_96_${pred_len} --model CycleNet --data Solar \
      --features M --seq_len 96 --pred_len $pred_len --enc_in 137 --cycle 144 \
      --cycle_mode lowrank --cycle_rank 4 \
      --model_type mlp --mrt_layers 1 --freq_layers 1 \
      --freq_loss_alpha 0.8 \
      --use_revin 0 \
      --train_epochs 30 --patience 15 --itr 1 --batch_size 64 \
      --learning_rate 0.01 --random_seed 2024 \
      > $log_dir/solar_lowrank4_mrt1_freq1_fredf08_${pred_len}.log 2>&1
done
echo "=== Solar MRT+FreqV1+FreDF+LowRank done ==="
