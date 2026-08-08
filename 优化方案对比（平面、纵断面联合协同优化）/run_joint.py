# -*- coding: utf-8 -*-
"""
run_joint.py — 优化方案对比主程序（平纵断面联合协同优化, pop=200, iter=1000）

【本实验的唯一方法：三维解空间(x,y,z)的平纵一体化协同优化】
  平面走向(每 10 m 控制点法向偏移 -> x,y)与纵断面坡度(每 10 m 变坡点高程 -> z)
  放入【同一决策向量】、在【同一次 IJS 寻优】中对 (x,y,z) 三维立体线形一起做
  全面彻底的搜索, 实现平面与纵断面的真正协同(区别于论文"先平面后纵断面"的分阶段
  串联)。求解桩号步长: 平面控制点 10 m、纵断面变坡点 10 m(用户指定, 全线加密;
  平面在 10 m 步长下受 R≥400m 约束的代价见 objective_joint 模块说明)。

三模式对比 (同一联合模型、同一约束下):
  M-A 现状方案(人工选线) : 实测平面中线(δ=0) + 人工粗放纵断面(0.5km 平滑地面线),
                            未做全局优化, 作为基线。
  M-B 单目标成本最优     : 平纵联合优化, 仅 min C (wC=1, wE=0)。
  M-C 平纵联合双目标(本文): 平纵联合优化, min C 与 min E 协同 + 熵权法客观决策。

  M-B → M-C 的差值 = "引入车流能耗协同优化"的净贡献。

数据: 数据.xlsx (北环高速实测)   公式: 林坤锐学位论文(式号见 objective*.py)
能耗单位: 全生命周期货币量(亿元, 与C同口径)   桥隧费用: 0(系数论文未给, 见 params/分析总结)
"""
import os, json, time, argparse, multiprocessing as mp
# 多进程下禁用 BLAS 内部多线程(每进程 1 核), 必须在 import numpy 之前设置。
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np

from params import ALGO, CASE, LONG_STD_100
from data_loader import load_alignment
from algorithms import run, VARIANTS
from objective import entropy_weights
from objective_joint import (make_plane_context, objectives_joint,
                             make_scalar_joint, decode_joint,
                             N_CTRL, M_PROF, M_PROF_VAR, CORRIDOR_HALF_W,
                             STEP_PLANE_M, STEP_PROFILE_M)
from safety import hazard_profile

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results"); os.makedirs(RESULTS, exist_ok=True)

POP_SIZE = ALGO["pop_size"]     # 200 (用户指定)
MAX_ITER = 1000                 # 联合优化迭代次数(用户指定 1000)
# 说明: 联合为单次 IJS, 总求值量 ≈ POP + 3·POP·iter = 200 + 3·200·1000 = 600,200,
#       恰与两阶段(Stage1+Stage2 各 iter=500)的 2·(200+3·200·500) = 600,400 大致相等,
#       即本设置下联合(iter=1000)与两阶段(iter=500)在【等总求值预算】下比较。

# ---- worker 进程内的全局(由 initializer 设定, 兼容 macOS spawn) ----
_PC = None
_CTX = None      # dict(C_ref, E_ref, pop0, lb, ub, max_iter)


def _init_worker(align, ctx):
    """worker 初始化: 重建 plane context(含 cKDTree)与共享寻优上下文, 每进程一次。"""
    global _PC, _CTX
    _PC = make_plane_context(align)
    _CTX = ctx


def _solve_one(task):
    """
    单个权重点的完整 IJS 联合寻优(在 worker 进程执行)。
    task = dict(tag, wC, wE, seed)
    结果与串行执行完全一致: 初始种群 pop0 与 seed 均由主进程固定下发, 与执行顺序无关。
    """
    pc, c = _PC, _CTX
    f = make_scalar_joint(pc, task["wC"], task["wE"], c["C_ref"], c["E_ref"])
    r = run(f, c["lb"], c["ub"], c["pop0"], c["max_iter"], task["seed"],
            **VARIANTS["V5_IJS"])
    C, E, pen, _ = objectives_joint(r["best_x"], pc)
    return dict(tag=task["tag"], wC=task["wC"], wE=task["wE"],
                C=float(C), E=float(E), pen=float(pen),
                best_x=r["best_x"].tolist(), curve=r["curve"].tolist())


def evaluate_joint(x, pc):
    """对联合决策向量计算四维指标 + 平面/纵断面线形序列。"""
    C, E, pen, info = objectives_joint(x, pc)
    d = decode_joint(x, pc)
    Q_series, Q_mean = hazard_profile(d["sta"], d["gz_new"], d["design_z"])
    return dict(C=C, E=E, penalty=pen, L_km=info["L_km"], Rmin=info["Rmin"],
                C_PING=info["C_PING"], C_TU=info["C_TU"], CR=info["CR"],
                CB=info["CB"], CS=info["CS"], CQ=info["CQ"],
                E_fuel=info["E_fuel"], E_ele=info["E_ele"],
                Vs=info["Vs"], Vh=info["Vh"], Q_mean=Q_mean,
                plane_x=d["xx"].tolist(), plane_y=d["yy"].tolist(),
                design_z=d["design_z"].tolist(), sta=d["sta"].tolist(),
                gz_new=d["gz_new"].tolist(), Q_series=Q_series.tolist())


def make_existing_x(pc, dim):
    """
    构造现状方案 M-A 的联合决策向量:
      平面: δ=0 (x[:N_CTRL]=0.5) -> 实测中线, 里程/走向不变;
      纵断面: 人工粗放设计线 = 沿实测中线地面高程的 0.5km 尺度平滑
              (未做全局精细优化), 在【变坡点】上反解为对应的 x[N_CTRL:]。
    依据: 现状为人工选线, 依实测地面线按较粗控制尺度布设纵断面(局部平滑、
          长直坡衔接), 故以 0.5km 平滑近似其未精细优化的纵断面。
    注: 决策变量是 M_PROF 个变坡点(10m 一个), 而 gz_new/sta 是 M_EVAL 个评价桩号
        (10m 一个); 故先在评价桩号上做 0.5km 平滑, 再采样到变坡点上反解。
    """
    x = np.full(dim, 0.5)                       # 平面 δ=0, 纵断面暂置零坡
    d0 = decode_joint(x, pc)
    gz = d0["gz_new"]; sta = d0["sta"]          # 评价桩号(10m)
    step = np.median(np.diff(sta))              # 评价桩号间距(≈10m)
    win = max(int(round(500.0 / step)), 3)      # 0.5km 平滑窗口
    if win % 2 == 0:
        win += 1
    kern = np.ones(win) / win
    design_A = np.convolve(gz, kern, mode="same")           # 平滑地面线(10m)
    # 采样到变坡点, 再反解为逐段纵坡决策量 x = 0.5*(i/imax + 1)
    design_A_ctrl = np.interp(d0["sta_ctrl"], sta, design_A)
    grades_A = np.diff(design_A_ctrl) / np.diff(d0["sta_ctrl"])
    x[N_CTRL:] = np.clip(
        0.5 * (grades_A / LONG_STD_100["grade_max"] + 1.0), 0.0, 1.0)
    return x


def main():
    global MAX_ITER
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="冒烟测试(iter=5, Pareto 仅 3 个权重点)")
    ap.add_argument("--workers", type=int, default=0,
                    help="并行进程数(默认: 按任务数与 CPU 核数自适应)")
    args = ap.parse_args()

    t0 = time.time()
    align = load_alignment()
    pc = make_plane_context(align)
    dim = N_CTRL + M_PROF_VAR
    lb, ub = np.zeros(dim), np.ones(dim)
    n_pareto = 21
    if args.smoke:
        MAX_ITER = 5
        n_pareto = 3
        print(f"[冒烟] iter={MAX_ITER}, Pareto 权重点={n_pareto}")
    print(f"[数据] 北环高速 {align['total_km']:.3f} km")
    print(f"[联合] 决策维度 dim={dim} (平面{N_CTRL} + 纵断面纵坡{M_PROF_VAR}), "
          f"走廊带±{CORRIDOR_HALF_W:.0f}m, pop={POP_SIZE}, iter={MAX_ITER}")

    # ---------- 熵权法权重(基准种群客观确定, 式5.3-5.4) ----------
    # 平面分量给足初始探索幅度(全走廊带), 避免平面子空间(仅 N_CTRL 维)在高维
    # 联合搜索中被纵断面(M_PROF_VAR 维)淹没。
    rng = np.random.default_rng(2025)
    base = np.empty((POP_SIZE, dim))
    base[:, :N_CTRL] = 0.5 + (rng.random((POP_SIZE, N_CTRL)) - 0.5) * 1.0
    base[:, N_CTRL:] = rng.random((POP_SIZE, M_PROF_VAR))
    base = np.clip(base, 0, 1)
    C0 = np.array([objectives_joint(base[i], pc)[0] for i in range(POP_SIZE)])
    E0 = np.array([objectives_joint(base[i], pc)[1] for i in range(POP_SIZE)])
    wC, wE = entropy_weights(C0, E0)
    C_ref, E_ref = float(C0.mean()), float(E0.mean())
    print(f"[熵权法] wC={wC:.4f}, wE={wE:.4f}")

    pop0 = base.copy()          # M-B/M-C/Pareto 共享同一初始种群保证公平

    # ---------- M-A 现状方案(人工选线, 未优化) ----------
    x_A = make_existing_x(pc, dim)
    res_A = evaluate_joint(x_A, pc)
    print(f"[M-A] C={res_A['C']/1e8:.4f}亿 E={res_A['E']/1e8:.4f}亿(全周期) "
          f"L={res_A['L_km']:.3f}km Q={res_A['Q_mean']:.3f}")

    # ---------- 并行求解 M-B / M-C / Pareto 权重扫描 ----------
    # 三者都是"同一初始种群 pop0、同一 seed=1000、只有权重不同"的独立 IJS 寻优,
    # 彼此无数据依赖, 故放到进程池里一起跑。每个任务的 pop0 与 seed 由主进程固定
    # 下发, 结果与串行执行逐位一致(仅执行顺序不同)。
    #   M-B: wC=1, wE=0            单目标成本最优
    #   M-C: wC,wE 为熵权法客观权重  本文方案
    #   Pareto: wC 从 0 到 1 扫描 n_pareto 个点(图C1 参考前沿 + 前沿变化趋势)
    w_grid = np.linspace(0.0, 1.0, n_pareto)
    tasks = ([dict(tag="M_B", wC=1.0, wE=0.0, seed=1000),
              dict(tag="M_C", wC=wC, wE=wE, seed=1000)]
             + [dict(tag=f"pareto_{k}", wC=float(w), wE=float(1 - w), seed=1000)
                for k, w in enumerate(w_grid)])
    ctx = dict(C_ref=C_ref, E_ref=E_ref, pop0=pop0, lb=lb, ub=ub,
               max_iter=MAX_ITER)
    n_workers = args.workers or min(len(tasks), max(1, (os.cpu_count() or 2) - 2))
    print(f"[并行] {len(tasks)} 个寻优任务 (M-B, M-C, Pareto×{n_pareto}), "
          f"{n_workers} 进程", flush=True)

    solved = {}
    with mp.Pool(n_workers, initializer=_init_worker,
                 initargs=(align, ctx)) as pool:
        for k, rec in enumerate(pool.imap_unordered(_solve_one, tasks), 1):
            solved[rec["tag"]] = rec
            el = time.time() - t0
            print(f"  [{k:2d}/{len(tasks)}] {rec['tag']:10s} wC={rec['wC']:.2f} "
                  f"C={rec['C']/1e8:.4f}亿 E={rec['E']/1e8:.4f}亿 pen={rec['pen']:.1e} "
                  f"| 用时{el/60:.1f}min ETA{el/k*(len(tasks)-k)/60:.1f}min", flush=True)

    # ---------- M-B 单目标成本最优 ----------
    rB = solved["M_B"]
    res_B = evaluate_joint(np.array(rB["best_x"]), pc)
    print(f"[M-B] C={res_B['C']/1e8:.4f}亿 E={res_B['E']/1e8:.4f}亿(全周期) "
          f"L={res_B['L_km']:.3f}km Rmin={res_B['Rmin']:.0f}m pen={res_B['penalty']:.2e}")

    # ---------- M-C 平纵联合双目标协同 (熵权法, 本文方案) ----------
    rC = solved["M_C"]
    res_C = evaluate_joint(np.array(rC["best_x"]), pc)
    print(f"[M-C] C={res_C['C']/1e8:.4f}亿 E={res_C['E']/1e8:.4f}亿(全周期) "
          f"L={res_C['L_km']:.3f}km Rmin={res_C['Rmin']:.0f}m pen={res_C['penalty']:.2e} "
          f"Q={res_C['Q_mean']:.3f}")

    # ---------- Pareto 权重扫描结果整理 ----------
    pareto_sweep = []
    for k, w1 in enumerate(w_grid):
        rec = solved[f"pareto_{k}"]
        pareto_sweep.append(dict(w1=float(w1), C=rec["C"], E=rec["E"],
                                 pen=rec["pen"]))
    # 图C1 沿用中间区间(0.1-0.9)作参考前沿, 保持与原图口径一致
    pareto = [p for p in pareto_sweep if 0.1 - 1e-9 <= p["w1"] <= 0.9 + 1e-9]
    entropy_point = dict(C=res_C["C"], E=res_C["E"], wC=wC, wE=wE)

    # 里程缩短(现状 M-A -> 本文 M-C)
    reduce_pct = (res_A["L_km"] - res_C["L_km"]) / res_A["L_km"] * 100
    print(f"[里程] 现状 {res_A['L_km']:.3f}km -> 联合优化 {res_C['L_km']:.3f}km "
          f"缩短 {reduce_pct:.2f}%")

    out = dict(
        meta=dict(dim=dim, N_ctrl=N_CTRL, M_prof=M_PROF, M_prof_var=M_PROF_VAR,
                  corridor_half_w=CORRIDOR_HALF_W, pop_size=POP_SIZE,
                  max_iter=MAX_ITER, wC=wC, wE=wE, C_ref=C_ref, E_ref=E_ref,
                  total_km=align["total_km"], Rmin_req=400,
                  step_plane_m=STEP_PLANE_M, step_profile_m=STEP_PROFILE_M,
                  n_pareto=n_pareto, smoke=bool(args.smoke),
                  n_workers=n_workers,
                  energy_unit="全生命周期元(亿元)",
                  note="平纵联合协同优化(三维解空间x/y/z): 平面控制点步长 "
                       f"{STEP_PLANE_M:.0f}m(受 R>=400m 约束), 纵断面变坡点步长 "
                       f"{STEP_PROFILE_M:.0f}m; 三方案 M-A/M-B/M-C 均在同一联合"
                       "模型下评估/寻优"),
        M_A=res_A, M_B=res_B, M_C=res_C,
        pareto=pareto, pareto_sweep=pareto_sweep, entropy_point=entropy_point,
        length_reduction_pct=reduce_pct,
        measured=dict(x=align["X"].tolist(), y=align["Y"].tolist()),
        convergence=rC["curve"], convergence_B=rB["curve"],
    )
    fn = "joint_results_smoke.json" if args.smoke else "joint_results.json"
    with open(os.path.join(RESULTS, fn), "w", encoding="utf-8") as fp:
        json.dump(out, fp, ensure_ascii=False, indent=2)
    print(f"[完成] {fn}  总耗时 {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
