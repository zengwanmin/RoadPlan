#!/usr/bin/env bash
# 问题15+16: 消融全量重跑(串行单进程, 保证测时可比) + 出图
set -u
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
echo "[$(date '+%F %T')] run_ablation 开始"
python3 -u run_ablation.py > ablation_p1516.log 2>&1 || { echo "ablation FAIL"; exit 1; }
echo "[$(date '+%F %T')] make_outputs 开始"
python3 -u make_outputs.py > make_outputs_p1516.log 2>&1 || { echo "make_outputs FAIL"; exit 1; }
echo "[$(date '+%F %T')] 消融链完成"
