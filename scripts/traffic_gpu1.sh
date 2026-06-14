#!/bin/bash
# Traffic GPU 1: Group B (MRT+FreqV1 lowrank6, all 4 preds)
cd /data/data_huaji/timeSeries/CycleNetBaseLine
log_dir=./logs/traffic
mkdir -p $log_dir

COMMON="--is_training 1 --root_path ./dataset/traffic/ --data_path traffic.csv --model CycleNet --data custom --features M --seq_len 96 --model_type mlp --use_revin 1 --enc_in 862 --cycle 168 --cycle_mode lowrank --cycle_rank 6 --mrt_layers 1 --freq_layers 1 --train_epochs 30 --patience 15 --itr 1 --batch_size 64 --learning_rate 0.002 --random_seed 2024"

echo "=== Group B: MRT+FreqV1 lowrank6 MSE ==="
for pred_len in 96 192 336 720; do
    echo "  B: MRT=1 FreqV1=1 lowrank6 pred=$pred_len"
    CUDA_VISIBLE_DEVICES=1 python -u run.py $COMMON \
      --model_id Traffic_96_${pred_len} --pred_len $pred_len \
      > $log_dir/traffic_lowrank6_mrt1_freq1_${pred_len}.log 2>&1
done

echo "=== GPU 1 done ==="
