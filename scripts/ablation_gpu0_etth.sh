#!/bin/bash
# ============================================================
# GPU 0: ETTh1 + ETTh2 消融 (MSE & FreDF)
# 预估: ~1.5h, 32 experiments
# ETTh1: cycle=24, enc_in=7, batch=256, lr=0.005, revin=1
# ETTh2: cycle=24, enc_in=7, batch=256, lr=0.005, revin=1
# ============================================================
set -e
cd /data/data_huaji/timeSeries/CycleNetBaseLine
GPU=0

COMMON="--is_training 1 --model CycleNet --seq_len 96 --model_type mlp \
  --train_epochs 30 --patience 15 --itr 1 --random_seed 2024 \
  --batch_size 256 --learning_rate 0.005 --use_revin 1"

run_ett() {
  local DATA=$1 LOGDIR=$2 PRED=$3 CYCLE=$4 EXTRA_FLAGS=$5 LOGNAME=$6
  local LOG="./logs/${LOGDIR}/${LOGNAME}.log"
  if [ -f "$LOG" ]; then
    echo "[SKIP] $LOG"
    return
  fi
  mkdir -p "./logs/${LOGDIR}"
  echo "[RUN] $LOGNAME  (GPU=$GPU)"
  CUDA_VISIBLE_DEVICES=$GPU python -u run.py \
    --data ${DATA} --root_path ./dataset/ETT-small/ --data_path ${DATA}.csv \
    --features M --enc_in 7 --cycle ${CYCLE} \
    --model_id ${LOGDIR}_96_${PRED} --pred_len ${PRED} \
    ${COMMON} ${EXTRA_FLAGS} \
    > ${LOG} 2>&1
}

echo "============ GPU $GPU: ETTh1 ============"
for PRED in 96 192 336 720; do
  # --- MSE ---
  run_ett ETTh1 etth1 $PRED 24 "" \
    "etth1_base_${PRED}"

  run_ett ETTh1 etth1 $PRED 24 "--mrt_layers 1" \
    "etth1_mrt1_${PRED}"

  # freq1 MSE: already done

  run_ett ETTh1 etth1 $PRED 24 "--mrt_layers 1 --freq_layers 1" \
    "etth1_mrt1_freq1_${PRED}"

  # --- FreDF (alpha=0.8) ---
  # base_fredf08: already done
  # freq1_fredf08: already done

  run_ett ETTh1 etth1 $PRED 24 "--mrt_layers 1 --freq_loss_alpha 0.8" \
    "etth1_mrt1_fredf08_${PRED}"

  run_ett ETTh1 etth1 $PRED 24 "--mrt_layers 1 --freq_layers 1 --freq_loss_alpha 0.8" \
    "etth1_mrt1_freq1_fredf08_${PRED}"
done

echo "============ GPU $GPU: ETTh2 ============"
for PRED in 96 192 336 720; do
  # --- MSE ---
  run_ett ETTh2 etth2 $PRED 24 "" \
    "etth2_base_${PRED}"

  run_ett ETTh2 etth2 $PRED 24 "--mrt_layers 1" \
    "etth2_mrt1_${PRED}"

  # freq1 MSE: already done

  run_ett ETTh2 etth2 $PRED 24 "--mrt_layers 1 --freq_layers 1" \
    "etth2_mrt1_freq1_${PRED}"

  # --- FreDF (alpha=0.8) ---
  # base_fredf08: already done
  # freq1_fredf08: already done

  run_ett ETTh2 etth2 $PRED 24 "--mrt_layers 1 --freq_loss_alpha 0.8" \
    "etth2_mrt1_fredf08_${PRED}"

  run_ett ETTh2 etth2 $PRED 24 "--mrt_layers 1 --freq_layers 1 --freq_loss_alpha 0.8" \
    "etth2_mrt1_freq1_fredf08_${PRED}"
done

echo "===== GPU $GPU ALL DONE ====="
