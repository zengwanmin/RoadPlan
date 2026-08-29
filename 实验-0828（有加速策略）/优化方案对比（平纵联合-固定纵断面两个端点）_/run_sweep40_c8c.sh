#!/usr/bin/env bash
set -u
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
python3 -u run_sweep40_c8c.py --workers 30 > sweep40_c8c.log 2>&1
