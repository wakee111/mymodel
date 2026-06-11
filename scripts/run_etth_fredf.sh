#!/bin/bash
# ETTh1 + ETTh2: FreDF (freq_loss_alpha=0.8) + FrequencyFilter V1
# GPU 0

cd /data/data_huaji/timeSeries/CycleNetBaseLine

COMMON="--freq_layers 1 --freq_loss_alpha 0.8 --model CycleNet --features M --seq_len 96 --model_type mlp --use_revin 1 --train_epochs 30 --patience 15 --itr 1 --batch_size 256 --learning_rate 0.005 --random_seed 2024"

for PRED in 96 192 336 720; do
    # ETTh1
    echo "=== ETTh1 freq1+FreDF pred=$PRED ==="
    CUDA_VISIBLE_DEVICES=0 python -u run.py \
        --is_training 1 \
        --root_path ./dataset/ETT-small/ \
        --data_path ETTh1.csv \
        --model_id ETTh1_96_${PRED} \
        --data ETTh1 \
        --pred_len ${PRED} \
        --enc_in 7 \
        --cycle 24 \
        ${COMMON} \
        > logs/etth1/etth1_freq1_fredf08_${PRED}.log 2>&1

    # ETTh2
    echo "=== ETTh2 freq1+FreDF pred=$PRED ==="
    CUDA_VISIBLE_DEVICES=0 python -u run.py \
        --is_training 1 \
        --root_path ./dataset/ETT-small/ \
        --data_path ETTh2.csv \
        --model_id ETTh2_96_${PRED} \
        --data ETTh2 \
        --pred_len ${PRED} \
        --enc_in 7 \
        --cycle 24 \
        ${COMMON} \
        > logs/etth2/etth2_freq1_fredf08_${PRED}.log 2>&1
done

echo "All ETTh1/ETTh2 FreDF experiments done!"
