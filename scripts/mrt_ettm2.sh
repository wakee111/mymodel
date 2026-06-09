#!/bin/bash
# ETTm2 MRT Ablation
# seq_len=96, cycle=96 (daily), enc_in=7, 15-min data
# MRT should excel here: seq=cycle=exactly one day, residual = "how today differs from average day"

model_name=CycleNet
root=./dataset/ETT-small/
dp=ETTm2.csv
mid=ETTm2
data_name=ETTm2
mt=mlp
sl=96
rs=2024

log_dir=./logs/mrt_ettm2
mkdir -p $log_dir

echo "=== ETTm2 MRT Ablation ==="

for pred_len in 96 192 336 720; do
    echo "--- pred_len=$pred_len ---"
    # Baseline
    echo "  [1/3] Baseline"
    python -u run.py \
      --is_training 1 --root_path $root --data_path $dp \
      --model_id ${mid}_${sl}_${pred_len} --model $model_name --data $data_name \
      --features M --seq_len $sl --pred_len $pred_len --enc_in 7 --cycle 96 \
      --model_type $mt --mrt_layers 0 \
      --train_epochs 30 --patience 5 --itr 1 --batch_size 256 \
      --learning_rate 0.005 --random_seed $rs \
      > $log_dir/mrt_base_${pred_len}.log 2>&1

    # MRT=1
    echo "  [2/3] +MRT=1"
    python -u run.py \
      --is_training 1 --root_path $root --data_path $dp \
      --model_id ${mid}_${sl}_${pred_len} --model $model_name --data $data_name \
      --features M --seq_len $sl --pred_len $pred_len --enc_in 7 --cycle 96 \
      --model_type $mt --mrt_layers 1 \
      --train_epochs 30 --patience 5 --itr 1 --batch_size 256 \
      --learning_rate 0.005 --random_seed $rs \
      > $log_dir/mrt_1_${pred_len}.log 2>&1

    # MRT=2
    echo "  [3/3] +MRT=2"
    python -u run.py \
      --is_training 1 --root_path $root --data_path $dp \
      --model_id ${mid}_${sl}_${pred_len} --model $model_name --data $data_name \
      --features M --seq_len $sl --pred_len $pred_len --enc_in 7 --cycle 96 \
      --model_type $mt --mrt_layers 2 \
      --train_epochs 30 --patience 5 --itr 1 --batch_size 256 \
      --learning_rate 0.005 --random_seed $rs \
      > $log_dir/mrt_2_${pred_len}.log 2>&1
done

echo "=== ETTm2 Done ==="
