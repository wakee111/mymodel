#!/bin/bash
# Traffic GPU 0: Group A (Baseline lookup 192/336/720) + Group E (MRT only lowrank6 96/192)
cd /data/data_huaji/timeSeries/CycleNetBaseLine
log_dir=./logs/traffic
mkdir -p $log_dir

COMMON="--is_training 1 --root_path ./dataset/traffic/ --data_path traffic.csv --model CycleNet --data custom --features M --seq_len 96 --model_type mlp --use_revin 1 --enc_in 862 --cycle 168 --train_epochs 30 --patience 15 --itr 1 --batch_size 64 --learning_rate 0.002 --random_seed 2024"

echo "=== Group A: Baseline lookup (complete missing preds) ==="
for pred_len in 192 336 720; do
    echo "  A: Baseline lookup pred=$pred_len"
    CUDA_VISIBLE_DEVICES=0 python -u run.py $COMMON \
      --model_id Traffic_96_${pred_len} --pred_len $pred_len \
      --cycle_mode lookup --mrt_layers 0 --freq_layers 0 \
      > $log_dir/traffic_baseline_lookup_${pred_len}.log 2>&1
done

echo "=== Group E: MRT only lowrank6 ==="
for pred_len in 96 192; do
    echo "  E: MRT=1 lowrank6 pred=$pred_len"
    CUDA_VISIBLE_DEVICES=0 python -u run.py $COMMON \
      --model_id Traffic_96_${pred_len} --pred_len $pred_len \
      --cycle_mode lowrank --cycle_rank 6 --mrt_layers 1 --freq_layers 0 \
      > $log_dir/traffic_lowrank6_mrt1_${pred_len}.log 2>&1
done

echo "=== GPU 0 done ==="
