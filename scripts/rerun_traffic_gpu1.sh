#!/bin/bash
# GPU 1: 重跑两个被中断的 Traffic LowRank4 实验
set -e
cd /data/data_huaji/timeSeries/CycleNetBaseLine
GPU=1

COMMON="--is_training 1 --model CycleNet --seq_len 96 --model_type mlp \
  --train_epochs 30 --patience 15 --itr 1 --random_seed 2024 \
  --learning_rate 0.002 --use_revin 1 --cycle_mode lowrank --cycle_rank 4"

run() {
  local PRED=$1 BS=$2 EXTRA_FLAGS=$3 LOGNAME=$4
  local LOG="./logs/traffic/${LOGNAME}.log"
  if [ -f "$LOG" ]; then
    echo "[SKIP] $LOG"
    return
  fi
  mkdir -p "./logs/traffic"
  echo "[RUN] $LOGNAME (GPU=$GPU, bs=$BS)"
  CUDA_VISIBLE_DEVICES=$GPU python -u run.py \
    --data custom --root_path ./dataset/traffic/ --data_path traffic.csv \
    --features M --enc_in 862 --cycle 168 \
    --model_id traffic_96_${PRED} --pred_len ${PRED} \
    ${COMMON} --batch_size ${BS} ${EXTRA_FLAGS} \
    > ${LOG} 2>&1
}

echo "=== GPU $GPU: re-run 2 incomplete Traffic logs ==="

# freq1_336: bs=64 no FreDF
run 336 64 "--freq_layers 1" traffic_lowrank4_freq1_336

# mrt1_freq1_720: bs=32 no FreDF
run 720 32 "--mrt_layers 1 --freq_layers 1" traffic_lowrank4_mrt1_freq1_720

echo "=== GPU $GPU done ==="
