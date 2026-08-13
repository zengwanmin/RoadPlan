#!/usr/bin/env bash
# 问题21新口径: 主实验全量重跑链(联合 -> 两阶段 -> 出图)
set -u
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
echo "[$(date '+%F %T')] run_joint 开始"
python3 -u run_joint.py --workers 23 > run_joint_p21.log 2>&1 || { echo "run_joint FAIL"; exit 1; }
echo "[$(date '+%F %T')] run_twostage 开始"
python3 -u run_twostage.py > run_twostage_p21.log 2>&1 || { echo "run_twostage FAIL"; exit 1; }
echo "[$(date '+%F %T')] make_outputs 开始"
python3 -u make_outputs.py > make_outputs_p21.log 2>&1 || { echo "make_outputs FAIL"; exit 1; }
echo "[$(date '+%F %T')] 主实验链完成"
