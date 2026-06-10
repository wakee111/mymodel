#!/bin/bash
# Weather MRT Multi-Seed Experiment
# Each pred_len log contains 5 seeds (2024-2028), patience=20

cd /data/data_huaji/timeSeries/CycleNetBaseLine
GPU=0
PRED_LENS=(96 192 336 720)
SEEDS=(2024 2025 2026 2027 2028)

for pred_len in "${PRED_LENS[@]}"; do
    LOG="logs/weather/weather_mrt1_${pred_len}_multiseed.log"
    echo "=== Weather MRT pred_len=${pred_len} multi-seed ===" | tee "$LOG"
    echo "Start time: $(date)" | tee -a "$LOG"

    for seed in "${SEEDS[@]}"; do
        echo "" | tee -a "$LOG"
        echo "==========================================" | tee -a "$LOG"
        echo "  SEED=${seed}  START: $(date)" | tee -a "$LOG"
        echo "==========================================" | tee -a "$LOG"

        CUDA_VISIBLE_DEVICES=$GPU python -u run.py \
            --is_training 1 \
            --root_path ./dataset/weather/ \
            --data_path weather.csv \
            --model_id weather_96_${pred_len} \
            --model CycleNet \
            --data custom \
            --features M \
            --seq_len 96 \
            --pred_len $pred_len \
            --enc_in 21 \
            --cycle 144 \
            --model_type mlp \
            --use_revin 1 \
            --mrt_layers 1 \
            --train_epochs 30 \
            --patience 20 \
            --itr 1 \
            --batch_size 256 \
            --learning_rate 0.005 \
            --random_seed $seed \
            >> "$LOG" 2>&1

        echo "  SEED=${seed}  END: $(date)" | tee -a "$LOG"
    done

    echo "" | tee -a "$LOG"
    echo "=== Weather MRT pred_len=${pred_len} ALL DONE ===" | tee -a "$LOG"
    echo "End time: $(date)" | tee -a "$LOG"
done

echo ""
echo "=== ALL WEATHER EXPERIMENTS COMPLETED ==="
