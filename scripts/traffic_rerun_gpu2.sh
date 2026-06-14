#!/bin/bash
# GPU 2: Rank4 D(FreqV1 only) + E(MRT only)
cd /data/data_huaji/timeSeries/CycleNetBaseLine
PY=/data/data_linsf/Anaconda3/envs/TimeSeries_huaji/bin/python
LOGD=./logs/traffic
mkdir -p $LOGD

S="--is_training 1 --root_path ./dataset/traffic/ --data_path traffic.csv --model CycleNet --data custom --features M --seq_len 96 --model_type mlp --use_revin 1 --enc_in 862 --cycle 168 --cycle_mode lowrank --cycle_rank 4 --train_epochs 30 --patience 15 --itr 1 --batch_size 64 --learning_rate 0.002 --random_seed 2024"

echo "== D rank4: FreqV1 only =="
for p in 96 192 336 720; do
  CUDA_VISIBLE_DEVICES=2 $PY -u run.py $S --model_id Traffic_96_${p} --pred_len $p \
    --freq_layers 1 --mrt_layers 0 > $LOGD/traffic_lowrank4_freq1_${p}.log 2>&1
done

echo "== E rank4: MRT only =="
for p in 96 192 336 720; do
  CUDA_VISIBLE_DEVICES=2 $PY -u run.py $S --model_id Traffic_96_${p} --pred_len $p \
    --mrt_layers 1 --freq_layers 0 > $LOGD/traffic_lowrank4_mrt1_${p}.log 2>&1
done

echo "GPU2 done"
