#!/bin/bash
# Config ④: MRT + Freq1 (3-stage training, FreDF only in Stage 3)
# Stage 1: Train MRT only (pure MSE)
# Stage 2: Freeze backbone+MRT, train Freq only (pure MSE)
# Stage 3: Unfreeze all, joint fine-tune + FreDF (α=0.8)
# Usage: bash scripts/run_solar_fredf_3stage.sh [gpu_id]  (default: 0)

GPU=${1:-0}
ROOT=/data/data_huaji/timeSeries/CycleNetBaseLine
PYTHON=/data/data_linsf/Anaconda3/envs/TimeSeries_huaji/bin/python

for pred_len in 96 192 336 720; do
    echo "=============================================="
    echo "Config ④ MRT+Freq1 3-stage + Stage3 FreDF | GPU=$GPU | pred_len=$pred_len"
    echo "=============================================="
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON scripts/multi_stage_train_solar.py \
        --pred_len $pred_len \
        --gpu 0 \
        --stage3_fredf \
        --seed 2024 \
        > logs/solar/solar_fredf_3stage_${pred_len}.log 2>&1
    echo "Done pred_len=$pred_len, exit code=$?"
done

echo ""
echo "Config ④ MRT+Freq1 3-stage + Stage3 FreDF all done!"
