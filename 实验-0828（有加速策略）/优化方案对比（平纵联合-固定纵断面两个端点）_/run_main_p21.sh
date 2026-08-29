#!/usr/bin/env bash
# 公平Pareto口径: 联合前沿 -> 两阶段前沿+公共熵权决策 -> 出图
set -u
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
# 最终口径：联合与两阶段均固定纵断面首末接线高程、无建筑密度约束、保留 OSM 交叉桥内生触发。
echo "[$(date '+%F %T')] run_joint 开始"
python3 -u run_joint.py --pareto 21 --workers 23 --fresh > run_joint_p21.log 2>&1 || { echo "run_joint FAIL"; exit 1; }
echo "[$(date '+%F %T')] run_twostage 开始"
python3 -u run_twostage.py --workers 23 --fresh > run_twostage_p21.log 2>&1 || { echo "run_twostage FAIL"; exit 1; }
echo "[$(date '+%F %T')] make_outputs 开始"
python3 -u make_outputs.py > make_outputs_p21.log 2>&1 || { echo "make_outputs FAIL"; exit 1; }
echo "[$(date '+%F %T')] 主实验链完成"
