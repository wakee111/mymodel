#!/bin/bash
# ============================================================
# GPU 0: Traffic LowRank4 消融 (仅剩 FreDF, 12 experiments)
# Traffic: cycle=168, enc_in=862, batch=64(720→32), lr=0.002, revin=1
# MSE 已全部完成, 只跑 FreDF base/mrt1/freq1
# ============================================================
set -e
cd /data/data_huaji/timeSeries/CycleNetBaseLine
GPU=0

COMMON="--is_training 1 --model CycleNet --seq_len 96 --model_type mlp \
  --train_epochs 30 --patience 15 --itr 1 --random_seed 2024 \
  --learning_rate 0.002 --use_revin 1"

LR4="--cycle_mode lowrank --cycle_rank 4"

run() {
  local PRED=$1 EXTRA_FLAGS=$2 LOGNAME=$3
  local LOG="./logs/traffic/${LOGNAME}.log"
  if [ -f "$LOG" ]; then
    echo "[SKIP] $LOG"
    return
  fi
  local BS=64
  [ "$PRED" = "720" ] && BS=32
  mkdir -p "./logs/traffic"
  echo "[RUN] $LOGNAME  (GPU=$GPU, bs=$BS)"
  CUDA_VISIBLE_DEVICES=$GPU python -u run.py \
    --data custom --root_path ./dataset/traffic/ --data_path traffic.csv \
    --features M --enc_in 862 --cycle 168 \
    --model_id traffic_96_${PRED} --pred_len ${PRED} \
    ${COMMON} --batch_size ${BS} ${LR4} ${EXTRA_FLAGS} \
    > ${LOG} 2>&1
}

echo "============ GPU $GPU: Traffic LowRank4 FreDF (alpha=0.8) ============"
for PRED in 96 192 336 720; do
  # base (MSE already done as traffic_lowrank4_{pred}.log, no "base")
  run $PRED "--freq_loss_alpha 0.8" \
    "traffic_lowrank4_fredf08_${PRED}"

  # mrt1
  run $PRED "--mrt_layers 1 --freq_loss_alpha 0.8" \
    "traffic_lowrank4_mrt1_fredf08_${PRED}"

  # freq1
  run $PRED "--freq_layers 1 --freq_loss_alpha 0.8" \
    "traffic_lowrank4_freq1_fredf08_${PRED}"

  # mrt1+freq1: already done
done

echo "===== GPU $GPU ALL DONE ====="
