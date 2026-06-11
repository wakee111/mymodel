#!/bin/bash
# Ablation: FreDF loss ONLY (no FrequencyFilter)
# ETTh1 + ETTh2, GPU 0
cd /data/data_huaji/timeSeries/CycleNetBaseLine

COMMON="--freq_layers 0 --freq_loss_alpha 0.8 --model CycleNet --features M --seq_len 96 --model_type mlp --use_revin 1 --train_epochs 30 --patience 15 --itr 1 --batch_size 256 --learning_rate 0.005 --random_seed 2024"

for PRED in 96 192 336 720; do
    for DS in ETTh1 ETTh2; do
        if [ "$DS" = "ETTh1" ]; then
            CYCLE=24; DATA_PATH="ETTh1.csv"
        else
            CYCLE=24; DATA_PATH="ETTh2.csv"
        fi
        LOG="logs/${DS,,}/${DS,,}_base_fredf08_${PRED}.log"
        echo "=== $DS base+FreDF pred=$PRED ==="
        CUDA_VISIBLE_DEVICES=0 python -u run.py \
            --is_training 1 --root_path ./dataset/ETT-small/ \
            --data_path ${DATA_PATH} --model_id ${DS}_96_${PRED} \
            --data ${DS} --pred_len ${PRED} --enc_in 7 --cycle ${CYCLE} \
            ${COMMON} > ${LOG} 2>&1
    done
done
echo "All ETTh ablation done!"
