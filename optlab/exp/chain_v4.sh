#!/bin/bash
# v4 全量(双层, 8种子×8核) -> 最优解 L-BFGS-B 精修, 链式执行
set -e
cd "$(dirname "$0")"
python3 -u run_bilevel.py --tag v4 --seeds 8 --pop 32 --iter 150 --workers 8
python3 -u refine.py --inp bilevel_v4.json --tag v4 --maxiter 400
echo "[chain done]"
