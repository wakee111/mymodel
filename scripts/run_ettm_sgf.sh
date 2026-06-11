#!/bin/bash
# SGF module only (no MRT, no FreDF) on ETTm1 + ETTm2
# GPU 1
cd /data/data_huaji/timeSeries/CycleNetBaseLine

COMMON="--sgf_layers 1 --model CycleNet --features M --seq_len 96 --model_type mlp --use_revin 1 --train_epochs 30 --patience 15 --itr 1 --batch_size 256 --learning_rate 0.005 --random_seed 2024"

for PRED in 96 192 336 720; do
    # ETTm1
    echo "=== ETTm1 SGF pred=$PRED ==="
    CUDA_VISIBLE_DEVICES=1 python -u run.py \
        --is_training 1 --root_path ./dataset/ETT-small/ \
        --data_path ETTm1.csv --model_id ETTm1_96_${PRED} --data ETTm1 \
        --pred_len ${PRED} --enc_in 7 --cycle 96 \
        --sgf_prior_path ./stability_priors/stability_ETTm1_cycle96_L96.npy \
        ${COMMON} > logs/ettm1/ettm1_sgf1_${PRED}.log 2>&1

    # ETTm2
    echo "=== ETTm2 SGF pred=$PRED ==="
    CUDA_VISIBLE_DEVICES=1 python -u run.py \
        --is_training 1 --root_path ./dataset/ETT-small/ \
        --data_path ETTm2.csv --model_id ETTm2_96_${PRED} --data ETTm2 \
        --pred_len ${PRED} --enc_in 7 --cycle 96 \
        --sgf_prior_path ./stability_priors/stability_ETTm2_cycle96_L96.npy \
        ${COMMON} > logs/ettm2/ettm2_sgf1_${PRED}.log 2>&1
done
echo "All ETTm SGF done!"
