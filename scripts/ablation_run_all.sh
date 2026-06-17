#!/bin/bash
# ============================================================
# 一键启动所有消融实验 (后台运行, GPU 0-3)
# 用法: bash scripts/ablation_run_all.sh
# 监控: tail -f logs/ablation_progress.log
# ============================================================
cd /data/data_huaji/timeSeries/CycleNetBaseLine

PROGRESS_LOG="./logs/ablation_progress.log"
echo "===== Ablation started at $(date) =====" > $PROGRESS_LOG

nohup bash scripts/ablation_gpu0_etth.sh >> $PROGRESS_LOG 2>&1 &
PID0=$!
echo "GPU 0 (ETTh1+ETTh2): PID=$PID0"

nohup bash scripts/ablation_gpu1_ettm_weather.sh >> $PROGRESS_LOG 2>&1 &
PID1=$!
echo "GPU 1 (ETTm+Weather): PID=$PID1"

nohup bash scripts/ablation_gpu2_electricity.sh >> $PROGRESS_LOG 2>&1 &
PID2=$!
echo "GPU 2 (Electricity): PID=$PID2"

nohup bash scripts/ablation_gpu3_traffic_lowrank4.sh >> $PROGRESS_LOG 2>&1 &
PID3=$!
echo "GPU 3 (Traffic LowRank4): PID=$PID3"

echo ""
echo "All launched. PIDs: $PID0 $PID1 $PID2 $PID3"
echo "Monitor: tail -f $PROGRESS_LOG"
echo "Check:  bash scripts/ablation_check_results.sh"

wait $PID0 $PID1 $PID2 $PID3
echo "===== ALL DONE at $(date) =====" >> $PROGRESS_LOG
