#!/bin/bash
# 方法矩阵: 4 法 × 2 种子, 走廊带 ±500, 8 核
set -e
cd "$(dirname "$0")"
FW=$(cat fixw.txt)
WEIGHTED="0.2,0.8,3044056942.569071,1606033907.890261"
CE_W="0.6545,0.3455,3044056942.569071,1606033907.890261"
CORRIDOR_HALF_W=500 python3 -u run_entropy_dp.py --tag m1push --seeds 2 --pop 24 --iter 250 --workers 2 --fixw "$FW" --warm entropy_dp_w500.json > m1.log 2>&1 &
CORRIDOR_HALF_W=500 N_MODE=80 python3 -u run_entropy_dp.py --tag m2modes80 --seeds 2 --pop 24 --iter 250 --workers 2 --fixw "$FW" --warm entropy_dp_w500.json > m2.log 2>&1 &
CORRIDOR_HALF_W=500 python3 -u run_entropy_dp.py --tag m3we08 --seeds 2 --pop 24 --iter 250 --workers 2 --fixw "$WEIGHTED" --warm entropy_dp_w500.json > m3.log 2>&1 &
CORRIDOR_HALF_W=500 python3 -u run_entropy_dp.py --tag m4ce --seeds 2 --pop 24 --iter 250 --workers 2 --fixw "$CE_W" --warm entropy_dp_w500.json > m4.log 2>&1 &
wait
echo "=== matrix done ==="
grep -h '最优F' m1.log m2.log m3.log m4.log
