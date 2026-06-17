#!/bin/bash
# ============================================================
# 检查消融实验结果，提取 MSE/MAE
# 用法: bash scripts/ablation_check_results.sh
# ============================================================
cd /data/data_huaji/timeSeries/CycleNetBaseLine

echo "============================================================"
echo "  消融实验结果汇总  $(date '+%Y-%m-%d %H:%M')"
echo "============================================================"

for DS in etth1 etth2 ettm1 ettm2 weather electricity traffic; do
  echo ""
  echo "--- $DS ---"
  printf "%-40s %12s %12s %s\n" "config" "mse" "mae" "status"
  printf "%-40s %12s %12s %s\n" "----------------------------------------" "------------" "------------" "------"

  for LOGFILE in ./logs/${DS}/*.log; do
    [ -f "$LOGFILE" ] || continue
    BASENAME=$(basename "$LOGFILE" .log)

    # Check if training completed
    if grep -q "Early stopping" "$LOGFILE" 2>/dev/null || grep -q "Training ended" "$LOGFILE" 2>/dev/null || grep -q "test.*mse" "$LOGFILE" 2>/dev/null; then
      STATUS="✅ done"
    elif grep -q "Epoch: 30" "$LOGFILE" 2>/dev/null; then
      STATUS="✅ done"
    else
      # Check if still running or crashed
      if grep -q "mse" "$LOGFILE" 2>/dev/null; then
        STATUS="⚠️ partial"
      else
        STATUS="🔄 running?"
      fi
    fi

    # Extract final test mse & mae
    MSE=$(grep -oP 'mse:\K[0-9]+\.[0-9]+' "$LOGFILE" 2>/dev/null | tail -1)
    MAE=$(grep -oP 'mae:\K[0-9]+\.[0-9]+' "$LOGFILE" 2>/dev/null | tail -1)

    if [ -z "$MSE" ]; then
      # Try alternative format: test shape ...
      MSE=$(grep -i 'test.*shape' "$LOGFILE" 2>/dev/null | head -1)
      [ -z "$MSE" ] && MSE="-"
    fi
    [ -z "$MAE" ] && MAE="-"

    printf "%-40s %12s %12s %s\n" "$BASENAME" "$MSE" "$MAE" "$STATUS"
  done
done

echo ""
echo "Done. Use 'grep -l \"mse\" logs/*/...log' to check specific files."
