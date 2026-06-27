#!/bin/bash
# ETTm1 Stage 2 only with unfrozen MLP (freeze_backbone=0)
# Freq + MLP trained jointly during Stage 2, MRT frozen

source /data/data_linsf/Anaconda3/etc/profile.d/conda.sh
conda activate TimeSeries_huaji
cd /data/data_huaji/timeSeries/CycleNetBaseLine

for pl in 96 192 336 720; do
    echo "========================================="
    echo "ETTm1 Stage2-only unfreeze pred_len=${pl}"
    echo "Started at: $(date)"
    echo "========================================="

    CUDA_VISIBLE_DEVICES=5 python -u scripts/multi_stage_train.py \
        --dataset ettm1 --pred_len ${pl} \
        --stage2_epochs 15 --stage2_patience 8 \
        --stage2_freeze_backbone 0 --stage2_only \
        > logs/ettm1/ETTm1_stage2only_unfreeze_${pl}.log 2>&1

    echo "Finished pred_len=${pl} at $(date)"
    echo ""
done

echo "All done."
