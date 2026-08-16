# -*- coding: utf-8 -*-
"""
run_main.py — 主实验(双层): 外层 IJS 搜平面, 内层 DP 全局解纵断面。

方案:
  M-A 现状方案   : 平面 δ=0(实测中线) + 实测路面高程(既有设计线)
  M-B 成本最优   : wC=1, wE=0
  M-C 本文方案   : 前沿熵权决策(可行 + 预算 C≤(1+tol)·C_A + 非支配 + 熵权)
  Pareto 前沿    : wC 从 0 到 1 扫描

规范合规的分工(见 bilevel.py 头部):
  纵断面类(纵坡上下界/坡差/端点接线) —— DP 状态空间构造保证, pen 恒为 0
  平面类(R≥400m / 建筑 Tier2 禁区)   —— 外层惩罚; Tier1 为软代价, 不进 pen

用法:
  python3 run_main.py --smoke                     # 冒烟(iter=3, 前沿3点)
  python3 run_main.py --corridor 1000 --pareto 9  # 正式
"""
import os, json, time, argparse, multiprocessing as mp
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np

import objective_joint as OJ
from objective_joint import make_plane_context, N_MODE
from data_loader import load_alignment
from algorithms import run, VARIANTS
from safety import hazard_profile
import bilevel as BL

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results"); os.makedirs(RESULTS, exist_ok=True)

BUDGET_TOL = 0.10           # 改扩建预算约束: C ≤ (1+tol)×现状成本
_PC = None
_CTX = None


def _init_worker(align, ctx, corridor, density_on):
    global _PC, _CTX
    OJ.set_corridor(corridor)
    OJ.set_density(density_on)
    _PC = make_plane_context(align)
    _CTX = ctx


def _solve_one(task):
    """单个权重点的外层 IJS 寻优(内层每次求值都调 DP 全局解纵断面)。"""
    pc, c = _PC, _CTX

    def make_f(ps):
        return BL.make_outer_f(pc, task["wC"], task["wE"],
                               c["C_ref"], c["E_ref"], pen_scale=ps)
    # 与主线一致的软->硬两阶段惩罚调度
    r = None
    pop = np.array(c["pop0"], float)
    for ps in c["pen_schedule"]:
        r = run(make_f(ps), np.zeros(N_MODE), np.ones(N_MODE), pop,
                c["max_iter"], task["seed"], **VARIANTS["V5_IJS"])
        pop = np.clip(r["best_x"] + 0.05 * np.random.default_rng(
            task["seed"]).standard_normal((c["pop"], N_MODE)), 0.0, 1.0)
        pop[0] = r["best_x"]
    ew = BL.dp_energy_weight(task["wC"], task["wE"], c["C_ref"], c["E_ref"])
    C, E, pen, info, z = BL.evaluate(r["best_x"], pc, ew)
    return dict(tag=task["tag"], wC=task["wC"], wE=task["wE"],
                C=float(C), E=float(E), pen=float(pen),
                modes=np.asarray(r["best_x"]).tolist(),
                curve=np.asarray(r["curve"]).tolist())


def full_record(modes, pc, wC, wE, C_ref, E_ref, z_override=None):
    """把外层解展开为完整工程指标(含线形序列), 供出表出图与合规复核。"""
    if z_override is None:
        ew = BL.dp_energy_weight(wC, wE, C_ref, E_ref)
        C, E, pen, info, z = BL.evaluate(modes, pc, ew)
    else:
        C, E, pen, info, z = BL.evaluate_existing(pc)
    d = BL.plane_terrain(modes, pc)
    design_z = np.interp(d["sta"], d["sta_ctrl"], z)
    Q_series, Q_mean = hazard_profile(d["sta"], d["gz_new"], design_z)
    g = np.diff(z) / np.diff(d["sta_ctrl"])
    return dict(
        C=float(C), E=float(E), penalty=float(pen), L_km=info["L_km"],
        Rmin=info["Rmin"], C_PING=info["C_PING"], C_TU=info["C_TU"],
        CR=info["CR"], CB=info["CB"], CS=info["CS"], CQ=info["CQ"],
        E_fuel=info["E_fuel"], E_ele=info["E_ele"], Vs=info["Vs"],
        Vh=info["Vh"], Q_mean=Q_mean, L_eco_km=info["L_eco_km"],
        L_ic_km=info["L_ic_km"], L_dense1_km=info["L_dense1_km"],
        L_dense2_km=info["L_dense2_km"], soft_dense1=info["soft_dense1"],
        grade_max_pct=float(np.abs(g).max() * 100),
        grade_min_pct=float(np.abs(g).min() * 100),
        dgrade_max=float(np.abs(np.diff(g)).max()),
        z_start=float(z[0]), z_end=float(z[-1]),
        plane_x=d["xx"].tolist(), plane_y=d["yy"].tolist(),
        sta=d["sta"].tolist(), design_z=design_z.tolist(),
        gz_new=d["gz_new"].tolist(), design_z_ctrl=np.asarray(z).tolist(),
        sta_ctrl=d["sta_ctrl"].tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--corridor", type=float, default=1000.0)
    ap.add_argument("--no-density", action="store_true")
    ap.add_argument("--pop", type=int, default=40)
    ap.add_argument("--iter", type=int, default=200)
    ap.add_argument("--pareto", type=int, default=9)
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()

    OJ.set_corridor(args.corridor)
    OJ.set_density(not args.no_density)
    n_pareto = args.pareto
    if args.smoke:
        args.pop, args.iter, n_pareto = 8, 3, 3

    t0 = time.time()
    align = load_alignment()
    pc = make_plane_context(align)
    print(f"[数据] 北环高速 {align['total_km']:.3f} km", flush=True)
    print(f"[双层] 外层平面 {N_MODE} 维 + 内层 DP 纵断面(全局最优, DZ="
          f"{__import__('dp_profile').DP_DZ} m), 走廊带±{args.corridor:.0f}m, "
          f"pop={args.pop}, iter={args.iter}, "
          f"密度约束={'ON' if OJ.DENSITY_ON else 'OFF'}", flush=True)

    pop0, wC, wE, C_ref, E_ref = BL.baseline(pc, args.pop)
    print(f"[熵权法] wC={wC:.4f}, wE={wE:.4f}", flush=True)

    CA, EA, penA, infoA, _ = BL.evaluate_existing(pc)
    res_A = full_record(np.full(N_MODE, 0.5), pc, wC, wE, C_ref, E_ref,
                        z_override=True)
    budget = (1.0 + BUDGET_TOL) * CA
    print(f"[M-A] C={CA/1e8:.4f}亿 E={EA/1e8:.4f}亿 L={infoA['L_km']:.3f}km "
          f"pen={penA:.1e} 预算上限={budget/1e8:.4f}亿", flush=True)

    w_grid = np.linspace(0.0, 1.0, n_pareto)
    tasks = ([dict(tag="M_B", wC=1.0, wE=0.0, seed=1000),
              dict(tag="M_C", wC=wC, wE=wE, seed=1000)]
             + [dict(tag=f"pareto_{k}", wC=float(w), wE=float(1 - w), seed=1000)
                for k, w in enumerate(w_grid)])
    ctx = dict(C_ref=C_ref, E_ref=E_ref, pop0=pop0, pop=args.pop,
               max_iter=args.iter, pen_schedule=(0.3, 3.0))
    nw = args.workers or min(len(tasks), max(1, (os.cpu_count() or 2) - 2))
    print(f"[并行] {len(tasks)} 个寻优任务, {nw} 进程", flush=True)

    solved = {}
    with mp.Pool(nw, initializer=_init_worker,
                 initargs=(align, ctx, OJ.CORRIDOR_HALF_W, OJ.DENSITY_ON)) as pool:
        for k, rec in enumerate(pool.imap_unordered(_solve_one, tasks), 1):
            solved[rec["tag"]] = rec
            el = (time.time() - t0) / 60
            print(f"  [{k:2d}/{len(tasks)}] {rec['tag']:10s} wC={rec['wC']:.2f} "
                  f"C={rec['C']/1e8:.4f}亿 E={rec['E']/1e8:.4f}亿 "
                  f"pen={rec['pen']:.1e} | 用时{el:.1f}min", flush=True)

    res_B = full_record(np.array(solved["M_B"]["modes"]), pc, 1.0, 0.0,
                        C_ref, E_ref)
    sweep = [dict(w1=float(w), **{k: solved[f"pareto_{k_}"][k]
                                  for k in ("C", "E", "pen")})
             for k_, w in enumerate(w_grid)]
    for k_, w in enumerate(w_grid):
        sweep[k_]["modes"] = solved[f"pareto_{k_}"]["modes"]

    # ---- M-C: 前沿熵权决策(可行 + 预算 + 非支配 + 熵权) ----
    cands = [p for p in sweep if p["pen"] <= 1e-6 and p["C"] <= budget]
    if not cands:
        print("[告警] 预算内无可行前沿点, 回退到惩罚最小者", flush=True)
        cands = sorted(sweep, key=lambda p: p["pen"])[:1]
    nd = [p for p in cands
          if not any(q["C"] <= p["C"] and q["E"] <= p["E"]
                     and (q["C"] < p["C"] or q["E"] < p["E"]) for q in cands)]
    fw_C, fw_E = BL.entropy_weights([p["C"] for p in nd], [p["E"] for p in nd])
    C_arr = np.array([p["C"] for p in nd]); E_arr = np.array([p["E"] for p in nd])
    def nrm(a):
        r = a.max() - a.min()
        return np.zeros_like(a) if r < 1e-30 else (a.max() - a) / r
    score = fw_C * nrm(C_arr) + fw_E * nrm(E_arr)
    best = nd[int(np.argmax(score))]
    print(f"[决策] 可行{len(cands)}/{len(sweep)} 非支配{len(nd)} "
          f"前沿熵权 wC={fw_C:.4f}/wE={fw_E:.4f} -> w1={best['w1']:.2f}", flush=True)
    res_C = full_record(np.array(best["modes"]), pc, wC, wE, C_ref, E_ref)
    print(f"[M-C] C={res_C['C']/1e8:.4f}亿({(res_C['C']/CA-1)*100:+.2f}%) "
          f"E={res_C['E']/1e8:.4f}亿({(res_C['E']/EA-1)*100:+.2f}%) "
          f"L={res_C['L_km']:.3f}km Rmin={res_C['Rmin']:.0f}m "
          f"pen={res_C['penalty']:.1e}", flush=True)
    print(f"[合规] 纵坡 {res_C['grade_min_pct']:.3f}%~{res_C['grade_max_pct']:.3f}% "
          f"(限 0.3%~4.0%) 坡差 {res_C['dgrade_max']:.5f} (限 {BL.DG_LIM:.3f}) "
          f"Tier2 {res_C['L_dense2_km']:.3f}km 端点 {res_C['z_start']:.3f}/"
          f"{res_C['z_end']:.3f}", flush=True)

    out = dict(
        meta=dict(method="双层: 外层IJS平面 + 内层DP纵断面(全局最优)",
                  outer_dim=N_MODE, corridor_half_w=args.corridor,
                  density_on=OJ.DENSITY_ON, pop=args.pop, max_iter=args.iter,
                  n_pareto=n_pareto, wC=wC, wE=wE, C_ref=C_ref, E_ref=E_ref,
                  budget=budget, budget_tol=BUDGET_TOL, smoke=bool(args.smoke),
                  dp_dz=__import__("dp_profile").DP_DZ, dg_lim=BL.DG_LIM,
                  total_km=align["total_km"]),
        M_A=res_A, M_B=res_B, M_C=res_C,
        pareto_sweep=[{k: v for k, v in p.items() if k != "modes"} for p in sweep],
        entropy_point=dict(w1=best["w1"], wC=fw_C, wE=fw_E),
        convergence={t: solved[t]["curve"] for t in ("M_B", "M_C")},
    )
    fn = f"main_dp_w{int(args.corridor)}{'_smoke' if args.smoke else ''}.json"
    with open(os.path.join(RESULTS, fn), "w", encoding="utf-8") as fp:
        json.dump(out, fp, ensure_ascii=False, indent=1)
    print(f"[完成] {fn}  总耗时 {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
