#!/bin/bash
# finalexp 主矩阵: 6宽度×2口径 联合(各3种子) + ±500 两阶段×2口径
# 并发: 12×3 + 2 = 38 进程(留足 8 核余量且避让他人任务)
set -e
cd "$(dirname "$0")"
FWA=$(cat fixw_avg.txt); FWS=$(cat fixw_single.txt)
for W in 500 600 700 800 900 1000; do
  for MODE in avg single; do
    FW=$FWA; [ "$MODE" = "single" ] && FW=$FWS
    CORRIDOR_HALF_W=$W E_DIRECTION=$MODE \
      python3 -u run_entropy_dp.py --tag ${MODE}_w${W} --seeds 3 --pop 24 \
      --iter 200 --workers 3 --fixw "$FW" --warm warm_w${W}.json \
      > log_${MODE}_w${W}.log 2>&1 &
  done
done
for MODE in avg single; do
  FW=$FWA; [ "$MODE" = "single" ] && FW=$FWS
  CORRIDOR_HALF_W=500 E_DIRECTION=$MODE \
    python3 -u run_twostage_dp.py --tag ${MODE}_w500 --seeds 3 --pop 24 \
    --iter 200 --fixw "$FW" --warm warm_w500.json \
    > log_ts_${MODE}.log 2>&1 &
done
wait
echo "=== finalexp matrix done ==="
grep -h '最优F' log_avg_*.log log_single_*.log | head -12
grep -h '两阶段' log_ts_*.log
