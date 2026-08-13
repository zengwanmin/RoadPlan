#!/usr/bin/env bash
# 敏感性分析全量重跑(新口径 + E_ext 项目⑨) + 出图
set -u
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
echo "[$(date '+%F %T')] run_reopt 开始"
python3 -u run_reopt.py --workers 30 > reopt_p21.log 2>&1 || { echo "reopt FAIL"; exit 1; }
echo "[$(date '+%F %T')] make_outputs 开始"
python3 -u make_outputs.py > make_outputs_p21.log 2>&1 || { echo "make_outputs FAIL"; exit 1; }
echo "[$(date '+%F %T')] 敏感性链完成"
