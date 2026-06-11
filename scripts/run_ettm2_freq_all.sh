#!/bin/bash
# ETTm2: FrequencyFilter V1, V2, V3, V4 (all versions)
# GPU 1, pred_len = 96, 192, 336, 720

cd /data/data_huaji/timeSeries/CycleNetBaseLine

DATASET="ETTm2"
DATA_PATH="ETTm2.csv"
ENC_IN=7
CYCLE=96
USE_REVIN=1
BATCH=256
LR=0.005
EPOCHS=30
PATIENCE=15
SEED=2024

for PRED in 96 192 336 720; do
    for FREQ_VER in v1 v2 v3 v4; do
        if [ "$FREQ_VER" = "v1" ]; then
            FREQ_ARG="--freq_layers 1"
            SUFFIX="freq1"
        elif [ "$FREQ_VER" = "v2" ]; then
            FREQ_ARG="--freq_v2_layers 1"
            SUFFIX="freqv21"
        elif [ "$FREQ_VER" = "v3" ]; then
            FREQ_ARG="--freq_v3_layers 1"
            SUFFIX="freqv31"
        elif [ "$FREQ_VER" = "v4" ]; then
            FREQ_ARG="--freq_v4_layers 1"
            SUFFIX="freqv41"
        fi

        LOG="logs/ettm2/ettm2_${SUFFIX}_${PRED}.log"
        echo "=============================================="
        echo "Running: ETTm2 $SUFFIX pred=$PRED"
        echo "Log: $LOG"
        echo "=============================================="

        CUDA_VISIBLE_DEVICES=1 python -u run.py \
            --is_training 1 \
            --root_path ./dataset/ETT-small/ \
            --data_path ${DATA_PATH} \
            --model_id ${DATASET}_96_${PRED} \
            --model CycleNet \
            --data ${DATASET} \
            --features M \
            --seq_len 96 \
            --pred_len ${PRED} \
            --enc_in ${ENC_IN} \
            --cycle ${CYCLE} \
            --model_type mlp \
            --use_revin ${USE_REVIN} \
            ${FREQ_ARG} \
            --train_epochs ${EPOCHS} \
            --patience ${PATIENCE} \
            --itr 1 \
            --batch_size ${BATCH} \
            --learning_rate ${LR} \
            --random_seed ${SEED} \
            > ${LOG} 2>&1

        echo "Finished: ETTm2 $SUFFIX pred=$PRED (exit code: $?)"
    done
done

echo "All ETTm2 freq experiments done!"
