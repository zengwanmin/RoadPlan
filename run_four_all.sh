#!/usr/bin/env bash
# ============================================================================
# run_four_all.sh — 一键跑完 RoadPlan 四个实验
#
# 启动 Stage1(多算法对比 + 消融)后, 交给已有的 run_all_stages.sh 等待 Stage1
# 结束并顺序跑 Stage2(优化方案对比) / Stage3(敏感性分析), 每阶段末自动出图。
#
# 注意: 优化方案对比已按用户要求改为【平面/纵断面/评价全 10m, dim=4494】,
#       故 Stage2 比脚本注释里的 dim=114/~17min 慢很多(联合 iter=1000 约 55min),
#       且平面在 10m 步长下会塌缩(Rmin=0、里程增大) —— 这是 10m 的固有代价, 已确认接受。
#       其余三个实验用各自目录下的原配置(敏感性分析 dim=114、多算法/消融为纵断面口径),
#       不受该 10m 改动影响。
#
# 用法: bash run_four_all.sh   (建议后台运行, 总耗时约 6+ 小时)
# ============================================================================
set -u
cd "$(dirname "$0")"
ROOT="$(pwd)"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

echo "[$(date '+%F %T')] Stage1 启动: 多算法对比(6进程) + 消融(1进程, 串行)"
( cd "多算法对比" && setsid python3 -u run_comparison.py > comparison.log 2>&1 < /dev/null ) &
PC=$!
( cd "消融实验" && setsid python3 -u run_ablation.py > ablation.log 2>&1 < /dev/null ) &
PA=$!
echo "[$(date '+%F %T')] comparison pid=$PC, ablation pid=$PA -> 交给 run_all_stages.sh 编排 Stage2/3"
exec bash run_all_stages.sh "$PC" "$PA"
