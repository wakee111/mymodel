#!/bin/bash
# GPU 1: Rank4 B(MRT+FreqV1) + C(MRT+FreqV1+FreDF)
cd /data/data_huaji/timeSeries/CycleNetBaseLine
PY=/data/data_linsf/Anaconda3/envs/TimeSeries_huaji/bin/python
LOGD=./logs/traffic
mkdir -p $LOGD

S="--is_training 1 --root_path ./dataset/traffic/ --data_path traffic.csv --model CycleNet --data custom --features M --seq_len 96 --model_type mlp --use_revin 1 --enc_in 862 --cycle 168 --cycle_mode lowrank --cycle_rank 4 --train_epochs 30 --patience 15 --itr 1 --batch_size 64 --learning_rate 0.002 --random_seed 2024"

echo "== B rank4: MRT+FreqV1 =="
for p in 96 192 336 720; do
  CUDA_VISIBLE_DEVICES=1 $PY -u run.py $S --model_id Traffic_96_${p} --pred_len $p \
    --mrt_layers 1 --freq_layers 1 > $LOGD/traffic_lowrank4_mrt1_freq1_${p}.log 2>&1
done

echo "== C rank4: MRT+FreqV1+FreDF =="
for p in 96 192 336 720; do
  CUDA_VISIBLE_DEVICES=1 $PY -u run.py $S --model_id Traffic_96_${p} --pred_len $p \
    --mrt_layers 1 --freq_layers 1 --freq_loss_alpha 0.8 > $LOGD/traffic_lowrank4_mrt1_freq1_fredf08_${p}.log 2>&1
done

echo "GPU1 done"
