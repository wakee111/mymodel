#!/bin/bash
# ===========================================================================
# 3-Stage Training for ETTh1 with trend_freq mode (pure MSE)
#
# Stage 1: Train MRT only (serial, MRT=1, Freq=0) → save checkpoint
# Stage 2: Load MRT, switch to trend_freq, freeze backbone+MRT+cycle,
#           train Freq + alpha_t + alpha_h (3 epochs)
# Stage 3: Unfreeze all, joint fine-tune with low LR in trend_freq mode
#
# Usage:
#   bash scripts/multi_stage_train_etth1_tf.sh
#   bash scripts/multi_stage_train_etth1_tf.sh 96   # single pred_len
# ===========================================================================

set -e
set -o pipefail

ROOT=/data/data_huaji/timeSeries/CycleNetBaseLine
GPU=5  # GPU 5 has most free memory
PYTHON=/data/data_linsf/Anaconda3/envs/TimeSeries_huaji/bin/python
export CUDA_VISIBLE_DEVICES=${GPU}

cd $ROOT

PRED_LENS=(${1:-96 192 336 720})
SEED=2024

for PRED_LEN in "${PRED_LENS[@]}"; do
    echo ""
    echo "########################################################################"
    echo "#  ETTh1 3-Stage trend_freq  pred_len=${PRED_LEN}"
    echo "########################################################################"

    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOGFILE="logs/etth1/ETTh1_tf_3stage_pl${PRED_LEN}_${TIMESTAMP}.log"
    mkdir -p logs/etth1

    $PYTHON scripts/multi_stage_train_etth1_tf.py \
        --pred_len ${PRED_LEN} \
        --seed ${SEED} \
        --gpu 0 \
        2>&1 | tee ${LOGFILE}

    echo "[$(date)] pred_len=${PRED_LEN} done."
done

echo ""
echo "All ETTh1 trend_freq 3-stage experiments done."
