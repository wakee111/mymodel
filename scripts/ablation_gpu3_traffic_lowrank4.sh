#!/bin/bash
# ============================================================
# GPU 3: Traffic LowRank4 消融 (MSE + FreDF)
# 预估: ~7h, 21 experiments
# Traffic: cycle=168, enc_in=862, batch=64, lr=0.002, revin=1
# ============================================================
set -e
cd /data/data_huaji/timeSeries/CycleNetBaseLine
GPU=0

COMMON="--is_training 1 --model CycleNet --seq_len 96 --model_type mlp \
  --train_epochs 30 --patience 15 --itr 1 --random_seed 2024 \
  --learning_rate 0.002 --use_revin 1"

# LowRank4 base flags
LR4="--cycle_mode lowrank --cycle_rank 4"

run() {
  local PRED=$1 EXTRA_FLAGS=$2 LOGNAME=$3
  local LOG="./logs/traffic/${LOGNAME}.log"
  if [ -f "$LOG" ]; then
    echo "[SKIP] $LOG"
    return
  fi
  # batch_size=32 for pred=720 to avoid OOM, 64 for others
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

echo "============ GPU $GPU: Traffic LowRank4 MSE ============"
for PRED in 96 192 336 720; do
  # base lowrank4: already done

  # mrt1 lowrank4
  run $PRED "--mrt_layers 1" \
    "traffic_lowrank4_mrt1_${PRED}"

  # freq1 lowrank4 (only 720 missing, 96/192/336 done)
  if [ "$PRED" = "720" ]; then
    run $PRED "--freq_layers 1" \
      "traffic_lowrank4_freq1_${PRED}"
  fi

  # mrt1 + freq1 lowrank4
  run $PRED "--mrt_layers 1 --freq_layers 1" \
    "traffic_lowrank4_mrt1_freq1_${PRED}"
done

echo "============ GPU $GPU: Traffic LowRank4 FreDF (alpha=0.8) ============"
for PRED in 96 192 336 720; do
  # base lowrank4 FreDF
  run $PRED "--freq_loss_alpha 0.8" \
    "traffic_lowrank4_base_fredf08_${PRED}"

  # mrt1 lowrank4 FreDF
  run $PRED "--mrt_layers 1 --freq_loss_alpha 0.8" \
    "traffic_lowrank4_mrt1_fredf08_${PRED}"

  # freq1 lowrank4 FreDF
  run $PRED "--freq_layers 1 --freq_loss_alpha 0.8" \
    "traffic_lowrank4_freq1_fredf08_${PRED}"

  # mrt1+freq1 lowrank4 FreDF: already done
done

echo "===== GPU $GPU ALL DONE ====="
