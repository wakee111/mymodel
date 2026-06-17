#!/bin/bash
# ============================================================
# GPU 2: Electricity 消融 (MSE & FreDF)
# 预估: ~3.5h, 20 experiments
# Electricity: cycle=168, enc_in=321, batch=64, lr=0.005, revin=1
# ============================================================
set -e
cd /data/data_huaji/timeSeries/CycleNetBaseLine
GPU=2

COMMON="--is_training 1 --model CycleNet --seq_len 96 --model_type mlp \
  --train_epochs 30 --patience 15 --itr 1 --random_seed 2024 \
  --batch_size 64 --learning_rate 0.005 --use_revin 1"

run_elec() {
  local PRED=$1 EXTRA_FLAGS=$2 LOGNAME=$3
  local LOG="./logs/electricity/${LOGNAME}.log"
  if [ -f "$LOG" ]; then
    echo "[SKIP] $LOG"
    return
  fi
  mkdir -p "./logs/electricity"
  echo "[RUN] $LOGNAME  (GPU=$GPU)"
  CUDA_VISIBLE_DEVICES=$GPU python -u run.py \
    --data custom --root_path ./dataset/electricity/ --data_path electricity.csv \
    --features M --enc_in 321 --cycle 168 \
    --model_id electricity_96_${PRED} --pred_len ${PRED} \
    ${COMMON} ${EXTRA_FLAGS} \
    > ${LOG} 2>&1
}

echo "============ GPU $GPU: Electricity ============"
for PRED in 96 192 336 720; do
  # --- MSE ---
  # base: done | mrt1: done | freq1: done

  run_elec $PRED "--mrt_layers 1 --freq_layers 1" \
    "electricity_mrt1_freq1_${PRED}"

  # --- FreDF (alpha=0.8) ---
  # ALL FreDF standalone experiments missing (only LowRank versions exist)

  run_elec $PRED "--freq_loss_alpha 0.8" \
    "electricity_base_fredf08_${PRED}"

  run_elec $PRED "--mrt_layers 1 --freq_loss_alpha 0.8" \
    "electricity_mrt1_fredf08_${PRED}"

  run_elec $PRED "--freq_layers 1 --freq_loss_alpha 0.8" \
    "electricity_freq1_fredf08_${PRED}"

  run_elec $PRED "--mrt_layers 1 --freq_layers 1 --freq_loss_alpha 0.8" \
    "electricity_mrt1_freq1_fredf08_${PRED}"
done

echo "===== GPU $GPU ALL DONE ====="
