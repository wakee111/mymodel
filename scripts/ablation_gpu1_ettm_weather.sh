#!/bin/bash
# ============================================================
# GPU 1: ETTm1 + ETTm2 + Weather 消融 (MSE & FreDF)
# 预估: ~1.5h, 28 experiments
# ETTm1/m2: cycle=96, enc_in=7, batch=256, lr=0.005, revin=1
# Weather:  cycle=144, enc_in=21, batch=256, lr=0.005, revin=1
# ============================================================
set -e
cd /data/data_huaji/timeSeries/CycleNetBaseLine
GPU=1

COMMON_ETT="--is_training 1 --model CycleNet --seq_len 96 --model_type mlp \
  --train_epochs 30 --patience 15 --itr 1 --random_seed 2024 \
  --batch_size 256 --learning_rate 0.005 --use_revin 1"

COMMON_WTH="--is_training 1 --model CycleNet --seq_len 96 --model_type mlp \
  --train_epochs 30 --patience 15 --itr 1 --random_seed 2024 \
  --batch_size 256 --learning_rate 0.005 --use_revin 1"

run_ett() {
  local DATA=$1 LOGDIR=$2 PRED=$3 EXTRA_FLAGS=$4 LOGNAME=$5
  local LOG="./logs/${LOGDIR}/${LOGNAME}.log"
  if [ -f "$LOG" ]; then
    echo "[SKIP] $LOG"
    return
  fi
  mkdir -p "./logs/${LOGDIR}"
  echo "[RUN] $LOGNAME  (GPU=$GPU)"
  CUDA_VISIBLE_DEVICES=$GPU python -u run.py \
    --data ${DATA} --root_path ./dataset/ETT-small/ --data_path ${DATA}.csv \
    --features M --enc_in 7 --cycle 96 \
    --model_id ${LOGDIR}_96_${PRED} --pred_len ${PRED} \
    ${COMMON_ETT} ${EXTRA_FLAGS} \
    > ${LOG} 2>&1
}

run_wth() {
  local PRED=$1 EXTRA_FLAGS=$2 LOGNAME=$3
  local LOG="./logs/weather/${LOGNAME}.log"
  if [ -f "$LOG" ]; then
    echo "[SKIP] $LOG"
    return
  fi
  mkdir -p "./logs/weather"
  echo "[RUN] $LOGNAME  (GPU=$GPU)"
  CUDA_VISIBLE_DEVICES=$GPU python -u run.py \
    --data custom --root_path ./dataset/weather/ --data_path weather.csv \
    --features M --enc_in 21 --cycle 144 \
    --model_id weather_96_${PRED} --pred_len ${PRED} \
    ${COMMON_WTH} ${EXTRA_FLAGS} \
    > ${LOG} 2>&1
}

# ===================== ETTm1 =====================
echo "============ GPU $GPU: ETTm1 ============"
for PRED in 96 192 336 720; do
  # --- MSE: base only 96/192 missing ---
  if [ "$PRED" = "96" ] || [ "$PRED" = "192" ]; then
    run_ett ETTm1 ettm1 $PRED "" "ettm1_base_${PRED}"
  fi
  # mrt1 MSE: done | freq1 MSE: done

  # mrt1_freq1 MSE: only 336/720 missing
  if [ "$PRED" = "336" ] || [ "$PRED" = "720" ]; then
    run_ett ETTm1 ettm1 $PRED "--mrt_layers 1 --freq_layers 1" \
      "ettm1_mrt1_freq1_${PRED}"
  fi

  # --- FreDF (alpha=0.8) ---
  # base_fredf08: done | mrt1_fredf: done | freq1_fredf08: done

  run_ett ETTm1 ettm1 $PRED "--mrt_layers 1 --freq_layers 1 --freq_loss_alpha 0.8" \
    "ettm1_mrt1_freq1_fredf08_${PRED}"
done

# ===================== ETTm2 =====================
echo "============ GPU $GPU: ETTm2 ============"
for PRED in 96 192 336 720; do
  # --- MSE ---
  # base: done | mrt1: done | freq1: done

  run_ett ETTm2 ettm2 $PRED "--mrt_layers 1 --freq_layers 1" \
    "ettm2_mrt1_freq1_${PRED}"

  # --- FreDF (alpha=0.8) ---
  # base_fredf08: done | freq1_fredf08: done

  run_ett ETTm2 ettm2 $PRED "--mrt_layers 1 --freq_loss_alpha 0.8" \
    "ettm2_mrt1_fredf08_${PRED}"

  run_ett ETTm2 ettm2 $PRED "--mrt_layers 1 --freq_layers 1 --freq_loss_alpha 0.8" \
    "ettm2_mrt1_freq1_fredf08_${PRED}"
done

# ===================== Weather =====================
echo "============ GPU $GPU: Weather ============"
for PRED in 96 192 336 720; do
  # --- MSE ---
  # base: done | mrt1: done | freq1: done

  run_wth $PRED "--mrt_layers 1 --freq_layers 1" \
    "weather_mrt1_freq1_${PRED}"

  # --- FreDF (alpha=0.8) ---
  # fredf_base: done | fredf_mrt1: done | fredf_freq1: done

  run_wth $PRED "--mrt_layers 1 --freq_layers 1 --freq_loss_alpha 0.8" \
    "weather_fredf_mrt1_freq1_${PRED}"
done

echo "===== GPU $GPU ALL DONE ====="
