#!/bin/bash
# Config ③: Freq1 + FreDF (mrt=0, freq=1, freq_loss_alpha=0.8)
# Usage: bash scripts/run_solar_fredf_freq.sh [gpu_id]  (default: 0)

GPU=${1:-0}
ROOT=/data/data_huaji/timeSeries/CycleNetBaseLine
PYTHON=/data/data_linsf/Anaconda3/envs/TimeSeries_huaji/bin/python

for pred_len in 96 192 336 720; do
    echo "=============================================="
    echo "Config ③ Freq1+FreDF | GPU=$GPU | pred_len=$pred_len"
    echo "=============================================="
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON -u run.py \
        --is_training 1 \
        --model_id Solar_96_${pred_len} \
        --model CycleNet \
        --data Solar \
        --root_path ./dataset/Solar/ \
        --data_path solar_AL.txt \
        --features M \
        --seq_len 96 \
        --pred_len $pred_len \
        --enc_in 137 \
        --cycle 144 \
        --model_type mlp \
        --use_revin 0 \
        --mrt_layers 0 \
        --freq_layers 1 \
        --freq_loss_alpha 0.8 \
        --train_epochs 30 \
        --patience 15 \
        --itr 1 \
        --batch_size 64 \
        --learning_rate 0.01 \
        --random_seed 2024 \
        > logs/solar/solar_fredf_freq1_${pred_len}.log 2>&1
    echo "Done pred_len=$pred_len, exit code=$?"
done

echo ""
echo "Config ③ Freq1+FreDF all done!"
