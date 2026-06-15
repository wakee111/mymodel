#!/bin/bash
# ETTh2: MRT + Freq1 三阶段训练 + Stage3 FreDF
# Usage: bash scripts/run_etth2_fredf_3stage.sh [gpu_id]  (default: 0)

GPU=${1:-0}
ROOT=/data/data_huaji/timeSeries/CycleNetBaseLine
PYTHON=/data/data_linsf/Anaconda3/envs/TimeSeries_huaji/bin/python

for pred_len in 96 192 336 720; do
    echo "=============================================="
    echo "ETTh2 3-stage + Stage3 FreDF | GPU=$GPU | pred_len=$pred_len"
    echo "=============================================="
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON scripts/multi_stage_train.py \
        --dataset etth2 \
        --pred_len $pred_len \
        --gpu 0 \
        --stage3_fredf \
        --seed 2024 \
        > logs/etth2/etth2_fredf_3stage_${pred_len}.log 2>&1
    echo "Done pred_len=$pred_len, exit code=$?"
done

echo ""
echo "ETTh2 3-stage + Stage3 FreDF all done!"
