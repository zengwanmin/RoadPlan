#!/usr/bin/env bash
# 问题19: 多算法 PJ1-PJ6 联合规模阶梯全量 + 出图
set -u
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
echo "[$(date '+%F %T')] run_comparison(PJ) 开始"
python3 -u run_comparison.py --workers 30 > comparison_pj.log 2>&1 || { echo "comparison FAIL"; exit 1; }
echo "[$(date '+%F %T')] make_outputs 开始"
python3 -u make_outputs.py > make_outputs_pj.log 2>&1 || { echo "make_outputs FAIL"; exit 1; }
echo "[$(date '+%F %T')] 多算法链完成"
