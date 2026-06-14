#!/bin/bash
# GPU 0: Rank6 全部未完成
cd /data/data_huaji/timeSeries/CycleNetBaseLine
PY=/data/data_linsf/Anaconda3/envs/TimeSeries_huaji/bin/python
LOGD=./logs/traffic
mkdir -p $LOGD

S="--is_training 1 --root_path ./dataset/traffic/ --data_path traffic.csv --model CycleNet --data custom --features M --seq_len 96 --model_type mlp --use_revin 1 --enc_in 862 --cycle 168 --train_epochs 30 --patience 15 --itr 1 --batch_size 64 --learning_rate 0.002 --random_seed 2024"

echo "== A: baseline lookup 336/720 =="
for p in 336 720; do
  CUDA_VISIBLE_DEVICES=0 $PY -u run.py $S --model_id Traffic_96_${p} --pred_len $p \
    --cycle_mode lookup --mrt_layers 0 --freq_layers 0 > $LOGD/traffic_baseline_lookup_${p}.log 2>&1
done

echo "== E: MRT only lowrank6 96/192/336/720 =="
for p in 96 192 336 720; do
  CUDA_VISIBLE_DEVICES=0 $PY -u run.py $S --model_id Traffic_96_${p} --pred_len $p \
    --cycle_mode lowrank --cycle_rank 6 --mrt_layers 1 --freq_layers 0 > $LOGD/traffic_lowrank6_mrt1_${p}.log 2>&1
done

echo "== B: MRT+FreqV1 lowrank6 336/720 =="
for p in 336 720; do
  CUDA_VISIBLE_DEVICES=0 $PY -u run.py $S --model_id Traffic_96_${p} --pred_len $p \
    --cycle_mode lowrank --cycle_rank 6 --mrt_layers 1 --freq_layers 1 > $LOGD/traffic_lowrank6_mrt1_freq1_${p}.log 2>&1
done

echo "== C: MRT+FreqV1+FreDF lowrank6 336/720 =="
for p in 336 720; do
  CUDA_VISIBLE_DEVICES=0 $PY -u run.py $S --model_id Traffic_96_${p} --pred_len $p \
    --cycle_mode lowrank --cycle_rank 6 --mrt_layers 1 --freq_layers 1 --freq_loss_alpha 0.8 > $LOGD/traffic_lowrank6_mrt1_freq1_fredf08_${p}.log 2>&1
done

echo "== D: FreqV1 only lowrank6 336/720 =="
for p in 336 720; do
  CUDA_VISIBLE_DEVICES=0 $PY -u run.py $S --model_id Traffic_96_${p} --pred_len $p \
    --cycle_mode lowrank --cycle_rank 6 --freq_layers 1 --mrt_layers 0 > $LOGD/traffic_lowrank6_freq1_${p}.log 2>&1
done

echo "GPU0 done"
