#!/bin/bash
# Solar MRT + Baseline
# seq=96, cycle=144 (daily), enc_in=137, 10-min data, use_revin=0

log_dir=./logs/mrt_solar
mkdir -p $log_dir

for pred_len in 96 192 336 720; do
    echo "=== pred_len=$pred_len ==="

    echo "  [1/2] Baseline"
    python -u run.py \
      --is_training 1 --root_path ./dataset/Solar/ --data_path solar_AL.txt \
      --model_id Solar_96_${pred_len} --model CycleNet --data Solar \
      --features M --seq_len 96 --pred_len $pred_len --enc_in 137 --cycle 144 \
      --model_type mlp --mrt_layers 0 --use_revin 0 \
      --train_epochs 30 --patience 5 --itr 1 --batch_size 64 \
      --learning_rate 0.01 --random_seed 2024 \
      > $log_dir/solar_base_${pred_len}.log 2>&1

    echo "  [2/2] +MRT=1"
    python -u run.py \
      --is_training 1 --root_path ./dataset/Solar/ --data_path solar_AL.txt \
      --model_id Solar_96_${pred_len} --model CycleNet --data Solar \
      --features M --seq_len 96 --pred_len $pred_len --enc_in 137 --cycle 144 \
      --model_type mlp --mrt_layers 1 --use_revin 0 \
      --train_epochs 30 --patience 5 --itr 1 --batch_size 64 \
      --learning_rate 0.01 --random_seed 2024 \
      > $log_dir/solar_mrt1_${pred_len}.log 2>&1
done
echo "=== Solar done ==="
