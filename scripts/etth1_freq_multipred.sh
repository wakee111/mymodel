#!/bin/bash
# ETTh1 FrequencyFilter multi-pred experiment
# freq_layers=1, patience=15, seed=2024
cd /data/data_huaji/timeSeries/CycleNetBaseLine

for pred_len in 96 192 336 720; do
    echo "=== ETTh1 freq1 pred_len=${pred_len} ==="
    CUDA_VISIBLE_DEVICES=0 python -u run.py \
        --is_training 1 \
        --root_path ./dataset/ETT-small/ \
        --data_path ETTh1.csv \
        --model_id ETTh1_96_${pred_len} \
        --model CycleNet \
        --data ETTh1 \
        --features M \
        --seq_len 96 \
        --pred_len $pred_len \
        --enc_in 7 \
        --cycle 24 \
        --model_type mlp \
        --use_revin 1 \
        --freq_layers 1 \
        --train_epochs 30 \
        --patience 15 \
        --itr 1 \
        --batch_size 256 \
        --learning_rate 0.005 \
        --random_seed 2024 \
        > logs/etth1/etth1_freq1_${pred_len}.log 2>&1
    echo "  Done: $(date)"
done

echo "=== ALL DONE ==="
