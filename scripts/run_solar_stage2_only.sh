#!/bin/bash
# Solar Stage 2 only: MRT → Freq (frozen backbone), no Stage 3
# Longer Stage 2 with more patience for proper convergence

source /data/data_linsf/Anaconda3/etc/profile.d/conda.sh
conda activate TimeSeries_huaji
cd /data/data_huaji/timeSeries/CycleNetBaseLine

for pl in 96 192 336 720; do
    echo "========================================="
    echo "Solar Stage2-only pred_len=${pl}"
    echo "Started at: $(date)"
    echo "========================================="

    CUDA_VISIBLE_DEVICES=5 python -u scripts/multi_stage_train.py \
        --dataset solar --pred_len ${pl} \
        --stage2_epochs 20 --stage2_patience 10 \
        --stage2_only \
        > logs/solar/solar_stage2only_${pl}.log 2>&1

    echo "Finished pred_len=${pl} at $(date)"
    echo ""
done

echo "All done."
