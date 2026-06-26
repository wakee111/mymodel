#!/bin/bash
# Solar --no_stage3a: Stage 2 → Stage 3b directly (skip Stage 3a)
# Fixes freeze-then-unfreeze distribution shift

cd /data/data_huaji/timeSeries/CycleNetBaseLine

for pl in 96 192 336 720; do
    echo "========================================="
    echo "Solar no_stage3a pred_len=${pl}"
    echo "Started at: $(date)"
    echo "========================================="

    CUDA_VISIBLE_DEVICES=5 python -u scripts/multi_stage_train.py \
        --dataset solar --pred_len ${pl} \
        --stage2_epochs 10 --stage2_patience 5 \
        --stage3b_epochs 25 --stage3b_patience 12 \
        --stage3b_freq_lr_mult 1.0 --stage3b_model_lr_mult 0.33 --stage3b_mrt_lr_mult 0.10 \
        --no_stage3a \
        > logs/solar/solar_nostage3a_${pl}.log 2>&1

    echo "Finished pred_len=${pl} at $(date)"
    echo ""
done

echo "All done."
