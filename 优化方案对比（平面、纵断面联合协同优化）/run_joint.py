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

from params import ALGO, CASE
from data_loader import load_alignment
from algorithms import run, VARIANTS
from objective import entropy_weights
from objective_joint import (make_plane_context, objectives_joint,
                             make_scalar_joint, decode_joint, joint_baseline,
                             run_ijs_two_phase, START_AMP_M,
                             DIM, N_MODE, M_PROF, CORRIDOR_HALF_W,
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
    def make_f(ps):
        return make_scalar_joint(pc, task["wC"], task["wE"], c["C_ref"],
                                 c["E_ref"], pen_scale=ps)
    r = run_ijs_two_phase(make_f, c["lb"], c["ub"], c["pop0"],
                          c["max_iter"], task["seed"])
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
                L_eco_km=info["L_eco_km"], L_ic_km=info["L_ic_km"],
                L_bridge_new=info["L_bridge_new"],
                L_tunnel_new=info["L_tunnel_new"],
                L_dense1_km=info["L_dense1_km"],
                L_dense2_km=info["L_dense2_km"],
                soft_dense1=info["soft_dense1"],
                dense_depth_max=info["dense_depth_max"],
                plane_x=d["xx"].tolist(), plane_y=d["yy"].tolist(),
                design_z=d["design_z"].tolist(), sta=d["sta"].tolist(),
                gz_new=d["gz_new"].tolist(), Q_series=Q_series.tolist())


def make_existing_x(pc, dim):
    """
    构造现状方案 M-A 的联合决策向量:
      平面: 模态系数全取 0.5 -> δ=0 -> 实测中线, 里程/走向不变;
      纵断面: 直接采用【实测路面高程】作为既有设计线(GPS 实测的就是现状道路的
              设计线), 按纵坡编码反解为起点高程偏移与各段纵坡。

    注: 地面高程现已改为真实地形 DEM, 对地形做平滑并不等于既有道路的设计线,
        故此处不再用"地面线平滑"近似现状纵断面, 而是直接用实测路面高程。
    """
    from params import LONG_STD_100
    x = np.full(dim, 0.5)                        # 平面 δ=0
    d0 = decode_joint(x, pc)
    sta_ctrl = d0["sta_ctrl"]; gz_ctrl = d0["gz_ctrl"]
    # δ=0 时评价桩号与实测里程一致, 直接按里程取实测路面高程
    z_road = np.interp(sta_ctrl, pc["s_meas"], pc["gz_meas"])
    x[N_MODE] = np.clip(0.5 + (z_road[0] - gz_ctrl[0]) / (2.0 * START_AMP_M), 0.0, 1.0)
    g = np.diff(z_road) / np.diff(sta_ctrl)      # 现状道路各段纵坡
    x[N_MODE + 1:] = np.clip(0.5 + g / (2.0 * LONG_STD_100["grade_max"]), 0.0, 1.0)
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
    dim = DIM
    lb, ub = np.zeros(dim), np.ones(dim)
    n_pareto = 21
    if args.smoke:
        MAX_ITER = 5
        n_pareto = 3
        print(f"[冒烟] iter={MAX_ITER}, Pareto 权重点={n_pareto}")
    print(f"[数据] 北环高速 {align['total_km']:.3f} km")
    print(f"[联合] 决策维度 dim={dim} (平面模态{N_MODE} + 纵断面{M_PROF}), "
          f"走廊带±{CORRIDOR_HALF_W:.0f}m, pop={POP_SIZE}, iter={MAX_ITER}")

    # ---------- 熵权法权重(基准种群客观确定, 式5.3-5.4) ----------
    # 由 joint_baseline 统一产出, 两阶段对照(run_twostage.py)共用同一组
    # (wC, wE, C_ref, E_ref), 使两种方法最小化同一个标量目标 F, 结果可直接比较。
    x_A = make_existing_x(pc, dim)
    base, wC, wE, C_ref, E_ref = joint_baseline(pc, POP_SIZE, x_seed=x_A)
    print(f"[熵权法] wC={wC:.4f}, wE={wE:.4f} (与两阶段对照共用)")

    pop0 = base.copy()          # M-B/M-C/Pareto 共享同一初始种群保证公平

    # ---------- M-A 现状方案(人工选线, 未优化) ----------
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

    # ---------- M-C 标量寻优点(初始种群熵权, 保留作对照) ----------
    rC = solved["M_C"]
    res_C_scalar = evaluate_joint(np.array(rC["best_x"]), pc)
    print(f"[M-C标量] C={res_C_scalar['C']/1e8:.4f}亿 E={res_C_scalar['E']/1e8:.4f}亿 "
          f"L={res_C_scalar['L_km']:.3f}km pen={res_C_scalar['penalty']:.2e}")

    # ---------- Pareto 权重扫描结果整理(含 best_x) ----------
    pareto_sweep = []
    for k, w1 in enumerate(w_grid):
        rec = solved[f"pareto_{k}"]
        pareto_sweep.append(dict(w1=float(w1), C=rec["C"], E=rec["E"],
                                 pen=rec["pen"], best_x=rec["best_x"]))

    # ---------- M-C 最终方案: 前沿熵权决策(论文第5章"先前沿、后决策"流程) ----------
    # ①候选 = 扫描解 + 标量寻优点, 要求可行(pen≈0);
    # ②预算约束: C ≤ (1+BUDGET_TOL)×现状成本(改扩建工程预算约束, 剔除
    #   "全线高架"等端部退化解, 避免熵权法对离群极差敏感);
    # ③非支配筛选; ④熵权法(式5.3-5.4, 极差标准化)加权得分取最大者。
    BUDGET_TOL = 0.10
    cands = [dict(tag=f"pareto_{k}", **p) for k, p in enumerate(pareto_sweep)]
    cands.append(dict(tag="M_C_scalar", w1=wC, C=res_C_scalar["C"],
                      E=res_C_scalar["E"], pen=res_C_scalar["penalty"],
                      best_x=rC["best_x"]))
    feas = [c for c in cands if c["pen"] <= 1e-6
            and c["C"] <= (1 + BUDGET_TOL) * res_A["C"]]
    if not feas:
        # 空集会让下面的 front[...] / M.min(0) 直接抛异常, 且看不出原因。
        # 密度 Tier2 这类硬约束收紧后确有可能全部候选不可行, 故降级并说明原因。
        pen_ok = [c for c in cands if c["pen"] <= 1e-6]
        pen_min = min((c["pen"] for c in cands), default=float("nan"))
        if pen_ok:
            print(f"[警告] 无候选同时满足 pen<=1e-6 与预算 C<=(1+{BUDGET_TOL})×现状; "
                  f"{len(pen_ok)} 个候选满足惩罚但超预算 -> 放宽预算约束降级选择",
                  flush=True)
            feas = pen_ok
        else:
            print(f"[警告] 无候选满足 pen<=1e-6(最小惩罚 {pen_min:.6g}) -> "
                  f"退化为按最小惩罚选择。请检查硬约束(密度 Tier2 / 最小半径)是否过严, "
                  f"以及走廊带是否被禁区封堵(见 building_density.corridor_passability)",
                  flush=True)
            feas = [c for c in cands if c["pen"] <= pen_min + 1e-12]
    front = [c for c in feas
             if not any((o["C"] <= c["C"] and o["E"] <= c["E"])
                        and (o["C"] < c["C"] or o["E"] < c["E"]) for o in feas)]
    M = np.array([[c["C"], c["E"]] for c in front])
    mn, mx = M.min(0), M.max(0)
    if np.all(mx - mn < 1e-6):                      # 退化: 前沿点全相同(冒烟等)
        w_front = np.array([0.5, 0.5])
        score = np.zeros(len(front))
    else:
        Zn = (mx - M) / (mx - mn + 1e-12)           # 成本型极差标准化
        P = (Zn + 1e-6) / (Zn + 1e-6).sum(0)
        Ej = -(P * np.log(P)).sum(0) / np.log(max(len(front), 2))   # 式5.3
        w_front = (1 - Ej) / ((1 - Ej).sum() + 1e-12)               # 式5.4
        score = Zn @ w_front
    sel = front[int(score.argmax())]
    print(f"[决策] 可行{len(feas)}/{len(cands)}, 前沿{len(front)}个, "
          f"前沿熵权 wC={w_front[0]:.4f}/wE={w_front[1]:.4f} "
          f"-> 选中 {sel['tag']}(w1={sel['w1']:.2f})")
    res_C = evaluate_joint(np.array(sel["best_x"]), pc)
    print(f"[M-C] C={res_C['C']/1e8:.4f}亿 E={res_C['E']/1e8:.4f}亿(全周期) "
          f"L={res_C['L_km']:.3f}km Rmin={res_C['Rmin']:.0f}m pen={res_C['penalty']:.2e} "
          f"Q={res_C['Q_mean']:.3f}")

    # 图C1 沿用中间区间(0.1-0.9)作参考前沿, 保持与原图口径一致
    pareto = [dict(w1=p["w1"], C=p["C"], E=p["E"], pen=p["pen"])
              for p in pareto_sweep if 0.1 - 1e-9 <= p["w1"] <= 0.9 + 1e-9]
    entropy_point = dict(C=res_C["C"], E=res_C["E"], wC=float(w_front[0]),
                         wE=float(w_front[1]), w1_selected=sel["w1"],
                         budget_tol=BUDGET_TOL)

    # 里程缩短(现状 M-A -> 本文 M-C)
    reduce_pct = (res_A["L_km"] - res_C["L_km"]) / res_A["L_km"] * 100
    print(f"[里程] 现状 {res_A['L_km']:.3f}km -> 联合优化 {res_C['L_km']:.3f}km "
          f"缩短 {reduce_pct:.2f}%")

    out = dict(
        meta=dict(dim=dim, n_mode=N_MODE, M_prof=M_PROF,
                  corridor_half_w=CORRIDOR_HALF_W, pop_size=POP_SIZE,
                  max_iter=MAX_ITER, wC=wC, wE=wE, C_ref=C_ref, E_ref=E_ref,
                  total_km=align["total_km"], Rmin_req=400,
                  step_plane_m=STEP_PLANE_M, step_profile_m=STEP_PROFILE_M,
                  n_pareto=n_pareto, smoke=bool(args.smoke),
                  n_workers=n_workers,
                  energy_unit="全生命周期元(亿元)",
                  note="平纵联合协同优化(准天然地面DEM口径): 立交7.8km常数计费"
                       "+桩号带土方豁免, 白云山隧道由生态区穿越长度内生, "
                       f"走廊带±{CORRIDOR_HALF_W:.0f}m; M-C=前沿熵权决策"
                       "(可行+预算约束C≤1.1×现状+非支配+熵权, 论文第5章流程)"),
        M_A=res_A, M_B=res_B, M_C=res_C, M_C_scalar=res_C_scalar,
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
