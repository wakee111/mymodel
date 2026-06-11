#!/bin/bash
# GPU 0: ETTh1 + ETTh2 baseline (lookup) + lowrank
# Sequential within each GPU
cd /data/data_huaji/timeSeries/CycleNetBaseLine
COMMON="--model CycleNet --features M --seq_len 96 --model_type mlp --use_revin 1 --train_epochs 30 --patience 15 --itr 1 --batch_size 256 --learning_rate 0.005 --random_seed 2024"

for PRED in 96 192 336 720; do
    echo "=== GPU0 ETTh1 lookup pred=$PRED ==="
    CUDA_VISIBLE_DEVICES=0 python -u run.py --is_training 1 --root_path ./dataset/ETT-small/ \
        --data_path ETTh1.csv --model_id ETTh1_96_${PRED} --data ETTh1 \
        --pred_len ${PRED} --enc_in 7 --cycle 24 --cycle_mode lookup \
        ${COMMON} > logs/etth1/etth1_lookup_${PRED}.log 2>&1

    echo "=== GPU0 ETTh2 lookup pred=$PRED ==="
    CUDA_VISIBLE_DEVICES=0 python -u run.py --is_training 1 --root_path ./dataset/ETT-small/ \
        --data_path ETTh2.csv --model_id ETTh2_96_${PRED} --data ETTh2 \
        --pred_len ${PRED} --enc_in 7 --cycle 24 --cycle_mode lookup \
        ${COMMON} > logs/etth2/etth2_lookup_${PRED}.log 2>&1
done
echo "GPU0 all done!"
