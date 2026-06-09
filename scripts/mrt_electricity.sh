#!/bin/bash
# Electricity MRT Ablation
# seq=96, cycle=168 (weekly), enc_in=321, hourly data
# seq/cycle = 96/168 = 0.57 (MRT should help if channel count allows)

log_dir=./logs/mrt_electricity
mkdir -p $log_dir

for pred_len in 96 192 336 720; do
    echo "=== pred_len=$pred_len ==="

    echo "  [1/3] Baseline"
    python -u run.py \
      --is_training 1 --root_path ./dataset/electricity/ --data_path electricity.csv \
      --model_id Electricity_96_${pred_len} --model CycleNet --data custom \
      --features M --seq_len 96 --pred_len $pred_len --enc_in 321 --cycle 168 \
      --model_type mlp --mrt_layers 0 \
      --train_epochs 30 --patience 5 --itr 1 --batch_size 64 \
      --learning_rate 0.005 --random_seed 2024 \
      > $log_dir/elec_base_${pred_len}.log 2>&1

    echo "  [2/3] +MRT=1"
    python -u run.py \
      --is_training 1 --root_path ./dataset/electricity/ --data_path electricity.csv \
      --model_id Electricity_96_${pred_len} --model CycleNet --data custom \
      --features M --seq_len 96 --pred_len $pred_len --enc_in 321 --cycle 168 \
      --model_type mlp --mrt_layers 1 \
      --train_epochs 30 --patience 5 --itr 1 --batch_size 64 \
      --learning_rate 0.005 --random_seed 2024 \
      > $log_dir/elec_mrt1_${pred_len}.log 2>&1

    echo "  [3/3] +MRT=2"
    python -u run.py \
      --is_training 1 --root_path ./dataset/electricity/ --data_path electricity.csv \
      --model_id Electricity_96_${pred_len} --model CycleNet --data custom \
      --features M --seq_len 96 --pred_len $pred_len --enc_in 321 --cycle 168 \
      --model_type mlp --mrt_layers 2 \
      --train_epochs 30 --patience 5 --itr 1 --batch_size 64 \
      --learning_rate 0.005 --random_seed 2024 \
      > $log_dir/elec_mrt2_${pred_len}.log 2>&1
done
echo "=== Electricity MRT done ==="
