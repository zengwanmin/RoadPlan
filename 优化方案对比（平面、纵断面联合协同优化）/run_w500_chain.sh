#!/usr/bin/env bash
# 固定端点 W500 正式重跑链: 联合主实验 -> 两阶段对照 -> C系列表图
# 严格按 README「正式运行顺序」第 1-3 步; 每步失败即中止, 不做静默拼接。
set -u
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

W=${W:-23}          # 联合阶段任务数 = M_B + M_C + Pareto×21 = 23

echo "[$(date '+%F %T')] STEP1 run_joint --corridor 500 --pareto 21 --workers $W --fresh"
python3 -u run_joint.py --corridor 500 --pareto 21 --workers "$W" --fresh \
    > w500_joint.log 2>&1 || { echo "STEP1 FAIL"; exit 1; }

echo "[$(date '+%F %T')] STEP2 run_twostage (绑定联合结果 SHA-256)"
python3 -u run_twostage.py --corridor 500 \
    --joint-result results/joint_results_w500_dens.json --fresh \
    > w500_twostage.log 2>&1 || { echo "STEP2 FAIL"; exit 1; }

echo "[$(date '+%F %T')] STEP3 make_outputs (来源指纹校验)"
python3 -u make_outputs.py > w500_make_outputs.log 2>&1 \
    || { echo "STEP3 FAIL"; exit 1; }

echo "[$(date '+%F %T')] W500 主链完成"
