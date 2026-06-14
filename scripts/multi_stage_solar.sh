#!/bin/bash
# Multi-stage training: MRT warmstart + Freq fine-tune → Solar all 4 pred_lens
cd /data/data_huaji/timeSeries/CycleNetBaseLine
PY=/data/data_linsf/Anaconda3/envs/TimeSeries_huaji/bin/python
LOGD=./logs/multi_stage_solar
mkdir -p $LOGD

for pred_len in 96 192 336 720; do
    echo "=== Multi-stage Solar pred=${pred_len} ==="
    CUDA_VISIBLE_DEVICES=0 $PY scripts/multi_stage_train_solar.py \
      --pred_len $pred_len --seed 2024 \
      > $LOGD/solar_multistage_mrt1_freq1_${pred_len}.log 2>&1
    echo "  pred=${pred_len} done"
done
echo "=== All done ==="
