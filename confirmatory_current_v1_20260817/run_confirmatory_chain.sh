#!/usr/bin/env bash
# 确认性实验及后续统计链(按 confirmatory README 顺序, 并发 48)
# 前置: 正式固定端点 W500 联合结果 + 两阶段对照 + make_outputs 已完成。
# 每步 fail-closed: 校验/分析失败即中止, 不静默继续。
set -u
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

W=${W:-48}      # 实测吞吐平台 13.4x, 48 并发取 ~96% 吞吐(见吞吐标定)

run() { echo "[$(date '+%F %T')] $*"; "$@" || { echo "FAIL: $*"; exit 1; }; }

# 1) 冒烟(仅验管线, 不作结论) + 校验
run python3 -u run_confirmatory.py --smoke --workers 2 --fresh
run python3 -u validate_results.py results/confirmatory_smoke.json

# 2) 正式 12算法 × 20配对重复 = 240 条, 精确 600k NFE
#    旧 108 条混合口径 partial 已归档于 Git 标签, 故用 --fresh 建立全新实验
run python3 -u run_confirmatory.py --workers "$W" --fresh
run python3 -u validate_results.py
run python3 -u analyze_confirmatory.py

# 3) 协作性(平面/纵断面结构消融)
run python3 -u run_collaboration.py --workers "$W"
run python3 -u validate_collaboration.py
run python3 -u analyze_collaboration.py

# 4) 稳健性(8 项预设重跑)
run python3 -u run_robustness.py --workers "$W"
run python3 -u validate_robustness.py
run python3 -u analyze_robustness.py

# 5) 运营情景 + 清单
run python3 -u evaluate_operational_scenarios.py
run python3 -u make_manifest.py

echo "[$(date '+%F %T')] 确认性链全部完成"
