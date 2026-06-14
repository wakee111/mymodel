#!/bin/bash
# Traffic rank4 全部实验 → GPU 1, nohup safe
cd /data/data_huaji/timeSeries/CycleNetBaseLine
PYTHON=/data/data_linsf/Anaconda3/envs/TimeSeries_huaji/bin/python
log_dir=./logs/traffic
mkdir -p $log_dir

COMMON="--is_training 1 --root_path ./dataset/traffic/ --data_path traffic.csv --model CycleNet --data custom --features M --seq_len 96 --model_type mlp --use_revin 1 --enc_in 862 --cycle 168 --cycle_mode lowrank --cycle_rank 4 --train_epochs 30 --patience 15 --itr 1 --batch_size 64 --learning_rate 0.002 --random_seed 2024"

echo "=== [rank4] E: MRT only ==="
for pred_len in 96 192 336 720; do
    echo "  rank4 MRT=1 pred=$pred_len"
    CUDA_VISIBLE_DEVICES=1 $PYTHON -u run.py $COMMON \
      --model_id Traffic_96_${pred_len} --pred_len $pred_len \
      --mrt_layers 1 --freq_layers 0 \
      > $log_dir/traffic_lowrank4_mrt1_${pred_len}.log 2>&1
done

echo "=== [rank4] D: FreqV1 only ==="
for pred_len in 96 192 336 720; do
    echo "  rank4 FreqV1=1 pred=$pred_len"
    CUDA_VISIBLE_DEVICES=1 $PYTHON -u run.py $COMMON \
      --model_id Traffic_96_${pred_len} --pred_len $pred_len \
      --freq_layers 1 --mrt_layers 0 \
      > $log_dir/traffic_lowrank4_freq1_${pred_len}.log 2>&1
done

echo "=== [rank4] B: MRT+FreqV1 ==="
for pred_len in 96 192 336 720; do
    echo "  rank4 MRT=1 FreqV1=1 pred=$pred_len"
    CUDA_VISIBLE_DEVICES=1 $PYTHON -u run.py $COMMON \
      --model_id Traffic_96_${pred_len} --pred_len $pred_len \
      --mrt_layers 1 --freq_layers 1 \
      > $log_dir/traffic_lowrank4_mrt1_freq1_${pred_len}.log 2>&1
done

echo "=== [rank4] C: MRT+FreqV1+FreDF ==="
for pred_len in 96 192 336 720; do
    echo "  rank4 MRT=1 FreqV1=1 FreDF(0.8) pred=$pred_len"
    CUDA_VISIBLE_DEVICES=1 $PYTHON -u run.py $COMMON \
      --model_id Traffic_96_${pred_len} --pred_len $pred_len \
      --mrt_layers 1 --freq_layers 1 --freq_loss_alpha 0.8 \
      > $log_dir/traffic_lowrank4_mrt1_freq1_fredf08_${pred_len}.log 2>&1
done

echo "=== GPU 1 rank4 all done ==="
