#!/bin/bash
# Traffic rank6 未完成实验 → GPU 0, nohup safe
cd /data/data_huaji/timeSeries/CycleNetBaseLine
PYTHON=/data/data_linsf/Anaconda3/envs/TimeSeries_huaji/bin/python
log_dir=./logs/traffic
mkdir -p $log_dir

COMMON="--is_training 1 --root_path ./dataset/traffic/ --data_path traffic.csv --model CycleNet --data custom --features M --seq_len 96 --model_type mlp --use_revin 1 --enc_in 862 --cycle 168 --train_epochs 30 --patience 15 --itr 1 --batch_size 64 --learning_rate 0.002 --random_seed 2024"

echo "=== [rank6] A: Baseline lookup 336/720 ==="
for pred_len in 336 720; do
    echo "  A: baseline lookup pred=$pred_len"
    CUDA_VISIBLE_DEVICES=0 $PYTHON -u run.py $COMMON \
      --model_id Traffic_96_${pred_len} --pred_len $pred_len \
      --cycle_mode lookup --mrt_layers 0 --freq_layers 0 \
      > $log_dir/traffic_baseline_lookup_${pred_len}.log 2>&1
done

echo "=== [rank6] E: MRT only lowrank6 96/192/336/720 ==="
for pred_len in 96 192 336 720; do
    echo "  E: MRT=1 lowrank6 pred=$pred_len"
    CUDA_VISIBLE_DEVICES=0 $PYTHON -u run.py $COMMON \
      --model_id Traffic_96_${pred_len} --pred_len $pred_len \
      --cycle_mode lowrank --cycle_rank 6 --mrt_layers 1 --freq_layers 0 \
      > $log_dir/traffic_lowrank6_mrt1_${pred_len}.log 2>&1
done

echo "=== [rank6] B: MRT+FreqV1 lowrank6 336/720 ==="
for pred_len in 336 720; do
    echo "  B: MRT=1 FreqV1=1 lowrank6 pred=$pred_len"
    CUDA_VISIBLE_DEVICES=0 $PYTHON -u run.py $COMMON \
      --model_id Traffic_96_${pred_len} --pred_len $pred_len \
      --cycle_mode lowrank --cycle_rank 6 --mrt_layers 1 --freq_layers 1 \
      > $log_dir/traffic_lowrank6_mrt1_freq1_${pred_len}.log 2>&1
done

echo "=== [rank6] C: MRT+FreqV1+FreDF lowrank6 336/720 ==="
for pred_len in 336 720; do
    echo "  C: MRT=1 FreqV1=1 FreDF(0.8) lowrank6 pred=$pred_len"
    CUDA_VISIBLE_DEVICES=0 $PYTHON -u run.py $COMMON \
      --model_id Traffic_96_${pred_len} --pred_len $pred_len \
      --cycle_mode lowrank --cycle_rank 6 --mrt_layers 1 --freq_layers 1 --freq_loss_alpha 0.8 \
      > $log_dir/traffic_lowrank6_mrt1_freq1_fredf08_${pred_len}.log 2>&1
done

echo "=== [rank6] D: FreqV1 only lowrank6 336/720 ==="
for pred_len in 336 720; do
    echo "  D: FreqV1=1 lowrank6 pred=$pred_len"
    CUDA_VISIBLE_DEVICES=0 $PYTHON -u run.py $COMMON \
      --model_id Traffic_96_${pred_len} --pred_len $pred_len \
      --cycle_mode lowrank --cycle_rank 6 --freq_layers 1 --mrt_layers 0 \
      > $log_dir/traffic_lowrank6_freq1_${pred_len}.log 2>&1
done

echo "=== GPU 0 rank6 all done ==="
