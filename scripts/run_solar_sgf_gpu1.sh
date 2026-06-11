#!/bin/bash
# Solar SGF seed=2025 on GPU 1
cd /data/data_huaji/timeSeries/CycleNetBaseLine
export PATH=/data/data_linsf/Anaconda3/envs/TimeSeries_huaji/bin:$PATH

COMMON="--sgf_layers 1 --model CycleNet --features M --seq_len 96 --model_type mlp --use_revin 0 --train_epochs 30 --patience 30 --itr 1 --batch_size 64 --learning_rate 0.01 --data Solar --root_path ./dataset/Solar/ --data_path solar_AL.txt --enc_in 137 --cycle 144 --sgf_prior_path ./stability_priors/stability_Solar_cycle144_L96.npy"

for PRED in 96 192 336 720; do
  echo "=== Solar SGF pred=$PRED seed=2025 ==="
  CUDA_VISIBLE_DEVICES=1 python -u run.py \
    --is_training 1 \
    --model_id Solar_96_${PRED} \
    --pred_len ${PRED} \
    --random_seed 2025 \
    ${COMMON} > logs/solar_sgf/solar_sgf1_${PRED}_seed2025.log 2>&1
done
echo "GPU 1 done!"
