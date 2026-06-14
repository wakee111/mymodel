#!/bin/bash
# Solar parallel fusion verification on GPU 1: pred_len 336/720.
cd /data/data_huaji/timeSeries/CycleNetBaseLine

PY=/data/data_linsf/Anaconda3/envs/TimeSeries_huaji/bin/python
LOG_DIR=./logs/parallel_solar
mkdir -p "$LOG_DIR"

COMMON="--is_training 1 --root_path ./dataset/Solar/ --data_path solar_AL.txt --model CycleNet --data Solar --features M --seq_len 96 --enc_in 137 --cycle 144 --model_type mlp --use_revin 0 --mrt_layers 1 --freq_layers 1 --fusion_mode parallel --train_epochs 30 --patience 30 --itr 1 --batch_size 64 --learning_rate 0.01 --random_seed 2024"

for pred_len in 336 720; do
  echo "=== Solar parallel pred_len=${pred_len} on GPU1 ==="
  CUDA_VISIBLE_DEVICES=1 "$PY" -u run.py \
    $COMMON \
    --model_id solar_96_${pred_len} \
    --pred_len ${pred_len} \
    > "$LOG_DIR/solar_mrt1_freq1_parallel_${pred_len}.log" 2>&1
done

echo "GPU1 Solar parallel verification done."
