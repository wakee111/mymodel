#!/bin/bash
# GPU 0: Rank6 all pending (12 runs) + Rank4 B (4 runs)
cd /data/data_huaji/timeSeries/CycleNetBaseLine
PY=/data/data_linsf/Anaconda3/envs/TimeSeries_huaji/bin/python
LD=./logs/traffic
mkdir -p $LD

S6="--is_training 1 --root_path ./dataset/traffic/ --data_path traffic.csv --model CycleNet --data custom --features M --seq_len 96 --model_type mlp --use_revin 1 --enc_in 862 --cycle 168 --train_epochs 30 --patience 15 --itr 1 --batch_size 64 --learning_rate 0.002 --random_seed 2024"

echo "=== [Rank6] A: baseline lookup 336/720 ==="
for p in 336 720; do
  CUDA_VISIBLE_DEVICES=0 $PY -u run.py $S6 --model_id Traffic_96_${p} --pred_len $p \
    --cycle_mode lookup --mrt_layers 0 --freq_layers 0 > $LD/traffic_baseline_lookup_${p}.log 2>&1
done

echo "=== [Rank6] E: MRT only lowrank6 96/192/336/720 ==="
for p in 96 192 336 720; do
  CUDA_VISIBLE_DEVICES=0 $PY -u run.py $S6 --model_id Traffic_96_${p} --pred_len $p \
    --cycle_mode lowrank --cycle_rank 6 --mrt_layers 1 --freq_layers 0 > $LD/traffic_lowrank6_mrt1_${p}.log 2>&1
done

echo "=== [Rank6] B: MRT+FreqV1 lowrank6 336/720 ==="
for p in 336 720; do
  CUDA_VISIBLE_DEVICES=0 $PY -u run.py $S6 --model_id Traffic_96_${p} --pred_len $p \
    --cycle_mode lowrank --cycle_rank 6 --mrt_layers 1 --freq_layers 1 > $LD/traffic_lowrank6_mrt1_freq1_${p}.log 2>&1
done

echo "=== [Rank6] C: MRT+FreqV1+FreDF lowrank6 336/720 ==="
for p in 336 720; do
  CUDA_VISIBLE_DEVICES=0 $PY -u run.py $S6 --model_id Traffic_96_${p} --pred_len $p \
    --cycle_mode lowrank --cycle_rank 6 --mrt_layers 1 --freq_layers 1 --freq_loss_alpha 0.8 > $LD/traffic_lowrank6_mrt1_freq1_fredf08_${p}.log 2>&1
done

echo "=== [Rank6] D: FreqV1 only lowrank6 336/720 ==="
for p in 336 720; do
  CUDA_VISIBLE_DEVICES=0 $PY -u run.py $S6 --model_id Traffic_96_${p} --pred_len $p \
    --cycle_mode lowrank --cycle_rank 6 --freq_layers 1 --mrt_layers 0 > $LD/traffic_lowrank6_freq1_${p}.log 2>&1
done

echo "=== [Rank4] B: MRT+FreqV1 lowrank4 ==="
for p in 96 192 336 720; do
  CUDA_VISIBLE_DEVICES=0 $PY -u run.py $S6 --model_id Traffic_96_${p} --pred_len $p \
    --cycle_mode lowrank --cycle_rank 4 --mrt_layers 1 --freq_layers 1 > $LD/traffic_lowrank4_mrt1_freq1_${p}.log 2>&1
done

echo "GPU0 ALL DONE"
