#!/usr/bin/env bash
# ============================================================================
# run_all_stages.sh — 四实验全量运行编排(适配 16 核 CPU 配额)
#
# 本机 nproc 报告 124 核, 但 cgroup 配额为 cpu.cfs_quota_us/period = 16 核。
# 按"总并发进程数 ≈ 配额的70%"(16 核 -> ~11 进程)分三阶段串行编排, 每阶段内部并行:
# (给满 16 会因争用反而更慢, 且给别的系统进程留余量)
#
#   Stage 1  多算法对比(6 进程, 每规模一个) + 消融(1 进程串行)   [已在外部启动]
#            —— 这两个实验把"平均运行时间"作为结果列上报(表A2/表B1),
#               故必须与其它实验隔离运行, 且规模内部串行, 保证耗时可比。
#   Stage 2  优化方案对比: run_joint(10 进程) + run_twostage(1 进程), dim=114, 实测 ~17.3 min
#   Stage 3  敏感性分析: run_reopt(9 进程, 226 个采样点重优化), dim=114, 实测 ~140 min
#
# 每阶段结束后立即生成该实验的 figures/ 与 tables/。
# 用法: nohup bash run_all_stages.sh > run_all_stages.log 2>&1 &
# ============================================================================
set -u
cd "$(dirname "$0")"
ROOT="$(pwd)"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

D_ABL="消融实验"
D_CMP="多算法对比"
D_OPT="优化方案对比（平面、纵断面联合协同优化）"
D_SEN="敏感性分析（平、纵联合，重优化）"

say() { echo "[$(date '+%F %T')] $*"; }

# 生成图表; 失败不中断整条链, 只记录
outputs() {
    local d="$1"
    say "  -> make_outputs: $d"
    ( cd "$ROOT/$d" && python3 make_outputs.py ) \
        && say "  -> OK 图表已生成: $d" \
        || say "  !! 失败 make_outputs: $d (见上方 traceback)"
}

# 等待某个 pid 结束(pid 可能已不存在)
wait_pid() {
    local p="$1" name="$2"
    while kill -0 "$p" 2>/dev/null; do sleep 30; done
    say "  $name (pid $p) 已结束"
}

# ---------------------------------------------------------------- Stage 1
# Stage 1 由外部先行启动, 这里只等它结束。传入两个 pid 作为参数。
if [ "$#" -ge 2 ]; then
    say "Stage 1: 等待 多算法对比(pid $1) 与 消融(pid $2) ..."
    wait_pid "$1" "多算法对比"
    wait_pid "$2" "消融"
else
    say "Stage 1: 未传入 pid, 按进程名等待 ..."
    while pgrep -f "[r]un_comparison\.py" >/dev/null || pgrep -f "[r]un_ablation\.py" >/dev/null; do
        sleep 30
    done
fi
say "Stage 1 完成"
[ -f "$ROOT/$D_ABL/results/ablation_results.json" ] \
    && outputs "$D_ABL" || say "  !! 缺 ablation_results.json, 跳过出图"
[ -f "$ROOT/$D_CMP/results/comparison_results.json" ] \
    && outputs "$D_CMP" || say "  !! 缺 comparison_results.json, 跳过出图"

# ---------------------------------------------------------------- Stage 2
# setsid + -u(无缓冲) + </dev/null: 曾观察到普通 nohup ... & 的 run_twostage.py
# 在无任何报错/OOM 迹象下于 Stage1 中途静默退出, 换成此写法后未再复现。
say "Stage 2: 启动 run_joint(10 进程) + run_twostage(1 进程, dim=114) ..."
( cd "$ROOT/$D_OPT" && setsid python3 -u run_joint.py --workers 10 > joint.log 2>&1 < /dev/null ) &
PJ=$!
( cd "$ROOT/$D_OPT" && setsid python3 -u run_twostage.py > twostage.log 2>&1 < /dev/null ) &
PT=$!
wait_pid "$PJ" "run_joint"
wait_pid "$PT" "run_twostage"
say "Stage 2 完成"
outputs "$D_OPT"

# ---------------------------------------------------------------- Stage 3
say "Stage 3: 启动 run_reopt(9 进程, 226 采样点, dim=114) ..."
( cd "$ROOT/$D_SEN" && setsid python3 -u run_reopt.py --workers 9 > reopt.log 2>&1 < /dev/null ) &
PR=$!
wait_pid "$PR" "run_reopt"
say "Stage 3 完成"
outputs "$D_SEN"

say "===== 全部阶段结束 ====="
say "结果文件:"
ls -la "$ROOT"/*/results/*.json 2>/dev/null | sed 's|'"$ROOT"'/||'
