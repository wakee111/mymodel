#!/bin/bash
# ETTm1 serial_progunfreeze: Multi-stage with progressive unfreeze (serial mode)
# Same config as solar_serial_progunfreeze

cd /data/data_huaji/timeSeries/CycleNetBaseLine

for pl in 96 192 336 720; do
    echo "========================================="
    echo "ETTm1 serial_progunfreeze pred_len=${pl}"
    echo "Started at: $(date)"
    echo "========================================="

    CUDA_VISIBLE_DEVICES=5 python -u scripts/multi_stage_train.py \
        --dataset ettm1 --pred_len ${pl} \
        --stage2_epochs 10 --stage2_patience 5 \
        --stage3a_epochs 10 --stage3a_patience 5 \
        --stage3b_epochs 25 --stage3b_patience 12 \
        --stage3b_freq_lr_mult 1.0 --stage3b_model_lr_mult 0.33 --stage3b_mrt_lr_mult 0.10 \
        > logs/ettm1/ETTm1_serial_progunfreeze_${pl}.log 2>&1

    echo "Finished pred_len=${pl} at $(date)"
    echo ""
done

echo "All done."
