#!/bin/bash
# Weather Ablation V2 — Modules BEFORE cycle removal (FIXED)
# Only reruns the 4 module configs (baseline unchanged, skip)

model_name=CycleNet
root_path_name=./dataset/weather/
data_path_name=weather.csv
model_id_name=weather
data_name=custom

model_type='mlp'
seq_len=96
random_seed=2024

log_dir=./logs/ablation_weather_v2
mkdir -p $log_dir

echo "============================================================"
echo "  Weather Ablation V2 — Modules BEFORE cycle (FIXED)"
echo "  seed=$random_seed  seq_len=$seq_len  cycle=144"
echo "============================================================"

for pred_len in 96 192 336 720
do
    echo ""
    echo "==== pred_len=$pred_len ===="

    # ---- MSC only ----
    echo "  [1/4] +MSC only (msc_layers=1)"
    python -u run.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 21 \
      --cycle 144 \
      --model_type $model_type \
      --msc_layers 1 \
      --freq_layers 0 \
      --channel_attn_layers 0 \
      --train_epochs 30 \
      --patience 5 \
      --itr 1 --batch_size 256 --learning_rate 0.005 --random_seed $random_seed \
      > $log_dir/weather_msc_${pred_len}.log 2>&1

    # ---- Freq only ----
    echo "  [2/4] +Freq only (freq_layers=1)"
    python -u run.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 21 \
      --cycle 144 \
      --model_type $model_type \
      --msc_layers 0 \
      --freq_layers 1 \
      --channel_attn_layers 0 \
      --train_epochs 30 \
      --patience 5 \
      --itr 1 --batch_size 256 --learning_rate 0.005 --random_seed $random_seed \
      > $log_dir/weather_freq_${pred_len}.log 2>&1

    # ---- CA only ----
    echo "  [3/4] +CA only (channel_attn_layers=1)"
    python -u run.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 21 \
      --cycle 144 \
      --model_type $model_type \
      --msc_layers 0 \
      --freq_layers 0 \
      --channel_attn_layers 1 \
      --train_epochs 30 \
      --patience 5 \
      --itr 1 --batch_size 256 --learning_rate 0.005 --random_seed $random_seed \
      > $log_dir/weather_ca_${pred_len}.log 2>&1

    # ---- ALL modules ----
    echo "  [4/4] ALL (msc+freq+ca)"
    python -u run.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 21 \
      --cycle 144 \
      --model_type $model_type \
      --msc_layers 1 \
      --freq_layers 1 \
      --channel_attn_layers 1 \
      --train_epochs 30 \
      --patience 5 \
      --itr 1 --batch_size 256 --learning_rate 0.005 --random_seed $random_seed \
      > $log_dir/weather_all_${pred_len}.log 2>&1

done

echo ""
echo "============================================================"
echo "  V2 complete!"
echo "============================================================"
