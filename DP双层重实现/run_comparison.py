# -*- coding: utf-8 -*-
"""
run_comparison.py — 多算法对比(双层): 6 种算法在【外层平面搜索】上的对比。

对比对象: IJS(本文) / JS / NSGA-II / GA / PSO / GWO
  前 5 者与 GWO 都在同一标量目标 F 上搜索; NSGA-II 在 (C,E) 双目标空间做非支配
  排序, 再用同一 F 从其前沿中取最优点, 保证与标量算法可比。

【为何本版可比性更强】纵断面由 DP 全局最优给出, 与外层算法无关, 因此各算法之间
的差异纯粹来自平面搜索能力; 且所有算法产出的解在纵坡/坡差/端点上都合规(pen 恒
为纵断面项 0, 只可能剩平面项 R/Tier2)。

规模阶梯 PJ1-PJ6 沿用"纵断面变坡点步长"定义(500/400/300/200/100/50 m):
  步长越细, DP 状态数越多、内层求解越慢, 但外层维度不变(平面 50 维)。
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
from benchmarks import run_GA, run_PSO, run_GWO, run_NSGA2
import bilevel as BL

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results"); os.makedirs(RESULTS, exist_ok=True)

SCALES = {"PJ1": 500.0, "PJ2": 400.0, "PJ3": 300.0,
          "PJ4": 200.0, "PJ5": 100.0, "PJ6": 50.0}
ALGOS = ["IJS", "JS", "NSGA-II", "GA", "PSO", "GWO"]
_PC = None
_CTX = None


def _init(align, corridor, density_on):
    global _PC
    OJ.set_corridor(corridor)
    OJ.set_density(density_on)
    _PC = None            # 每个 scale 需重建(步长变化), 在 _solve 内按需构建
    globals()["_ALIGN"] = align


def _solve(task):
    """(算法, 规模, 种子) 一个单元。"""
    global _PC
    algo, scale, seed = task["algo"], task["scale"], task["seed"]
    OJ.set_profile_step(SCALES[scale])          # 改变纵断面变坡点步长
    pc = make_plane_context(globals()["_ALIGN"])
    c = task
    lb, ub = np.zeros(N_MODE), np.ones(N_MODE)
    rng = np.random.default_rng(seed)
    pop0 = rng.random((c["pop"], N_MODE))
    pop0[0] = 0.5
    f = BL.make_outer_f(pc, c["wC"], c["wE"], c["C_ref"], c["E_ref"],
                        pen_scale=3.0)
    t0 = time.time()
    if algo == "IJS":
        r = run(f, lb, ub, pop0, c["iter"], seed, **VARIANTS["V5_IJS"])
        best_x, curve = r["best_x"], np.asarray(r["curve"])
    elif algo == "JS":
        r = run(f, lb, ub, pop0, c["iter"], seed, **VARIANTS["V1_JS"])
        best_x, curve = r["best_x"], np.asarray(r["curve"])
    elif algo == "GA":
        r = run_GA(f, lb, ub, pop0, c["iter"], seed)
        best_x, curve = r["best_x"], np.asarray(r["curve"])
    elif algo == "PSO":
        r = run_PSO(f, lb, ub, pop0, c["iter"], seed)
        best_x, curve = r["best_x"], np.asarray(r["curve"])
    elif algo == "GWO":
        r = run_GWO(f, lb, ub, pop0, c["iter"], seed)
        best_x, curve = r["best_x"], np.asarray(r["curve"])
    elif algo == "NSGA-II":
        fb = BL.make_outer_biobj(pc, c["wC"], c["wE"], c["C_ref"], c["E_ref"],
                                 pen_scale=3.0)
        r = run_NSGA2(fb, lb, ub, pop0, c["iter"], seed)
        X = np.atleast_2d(r["front_X"])
        # 用同一标量 F 从非支配前沿中取最优, 保证与标量算法同口径可比
        fs = np.array([f(x) for x in X])
        best_x, curve = X[int(np.argmin(fs))], np.array([fs.min()])
    else:
        raise ValueError(algo)
    wall = time.time() - t0
    best_f = float(f(best_x))
    ew = BL.dp_energy_weight(c["wC"], c["wE"], c["C_ref"], c["E_ref"])
    C, E, pen, info, z = BL.evaluate(best_x, pc, ew)
    d = BL.plane_terrain(best_x, pc)
    g = np.diff(z) / np.diff(d["sta_ctrl"])
    return dict(algo=algo, scale=scale, seed=seed, best_f=best_f,
                C=float(C), E=float(E), pen=float(pen),
                Rmin=float(info["Rmin"]), L_km=float(info["L_km"]),
                L_dense2_km=float(info["L_dense2_km"]),
                grade_max_pct=float(np.abs(g).max() * 100),
                dgrade_max=float(np.abs(np.diff(g)).max()),
                n_ctrl=int(len(d["sta_ctrl"])), wall_s=wall,
                curve=curve.tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--corridor", type=float, default=1000.0)
    ap.add_argument("--pop", type=int, default=40)
    ap.add_argument("--iter", type=int, default=150)
    ap.add_argument("--n_runs", type=int, default=5)
    ap.add_argument("--scales", default="PJ5",
                    help="逗号分隔, 如 PJ1,PJ5 或 all(六档全跑, 很贵)")
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()
    OJ.set_corridor(args.corridor)
    OJ.set_density(True)
    scales = list(SCALES) if args.scales == "all" else args.scales.split(",")
    if args.smoke:
        args.pop, args.iter, args.n_runs = 8, 3, 1
        scales = ["PJ5"]

    t0 = time.time()
    align = load_alignment()
    OJ.set_profile_step(SCALES["PJ5"])
    pc = make_plane_context(align)
    pop0, wC, wE, C_ref, E_ref = BL.baseline(pc, args.pop)
    print(f"[数据] 北环高速 {align['total_km']:.3f} km, 外层平面 {N_MODE} 维, "
          f"内层 DP 纵断面(全局最优)", flush=True)
    print(f"[熵权法] wC={wC:.4f}, wE={wE:.4f} (在 PJ5 基准下确定, 各档共用)",
          flush=True)
    print(f"[设置] {len(ALGOS)} 算法 × {len(scales)} 规模 × {args.n_runs} 种子, "
          f"pop={args.pop}, iter={args.iter}", flush=True)

    tasks = [dict(algo=a, scale=s, seed=1000 + k, pop=args.pop,
                  iter=args.iter, wC=wC, wE=wE, C_ref=C_ref, E_ref=E_ref)
             for s in scales for a in ALGOS for k in range(args.n_runs)]
    nw = args.workers or min(len(tasks), max(1, (os.cpu_count() or 2) - 2))
    print(f"[并行] {len(tasks)} 个单元, {nw} 进程", flush=True)

    recs = []
    with mp.Pool(nw, initializer=_init,
                 initargs=(align, OJ.CORRIDOR_HALF_W, OJ.DENSITY_ON)) as pool:
        for k, r in enumerate(pool.imap_unordered(_solve, tasks), 1):
            recs.append(r)
            print(f"  [{k:3d}/{len(tasks)}] {r['scale']} {r['algo']:8s} "
                  f"seed={r['seed']} F={r['best_f']:.5f} "
                  f"C={r['C']/1e8:.4f}亿 pen={r['pen']:.1e} "
                  f"[{r['wall_s']:.1f}s]", flush=True)

    # 汇总: 每 (规模, 算法) 的 F 均值/标准差, 并给出档内排名
    summary = {}
    for s in scales:
        rows = {}
        for a in ALGOS:
            fs = np.array([r["best_f"] for r in recs
                           if r["scale"] == s and r["algo"] == a])
            pens = np.array([r["pen"] for r in recs
                             if r["scale"] == s and r["algo"] == a])
            rows[a] = dict(F_mean=float(fs.mean()), F_std=float(fs.std()),
                           F_best=float(fs.min()),
                           feasible=int((pens <= 1e-6).sum()), n=int(len(fs)))
        order = sorted(ALGOS, key=lambda a: rows[a]["F_mean"])
        summary[s] = dict(rows=rows, rank=order)
        print(f"[{s}] 排名: " + " < ".join(order), flush=True)
        print(f"      IJS F={rows['IJS']['F_mean']:.5f} vs "
              f"次优 {order[1] if order[0]=='IJS' else order[0]}="
              f"{rows[order[1] if order[0]=='IJS' else order[0]]['F_mean']:.5f}",
              flush=True)

    out = dict(meta=dict(method="双层: 外层多算法平面搜索 + 内层DP纵断面",
                         outer_dim=N_MODE, corridor_half_w=args.corridor,
                         pop=args.pop, max_iter=args.iter, n_runs=args.n_runs,
                         scales={s: SCALES[s] for s in scales},
                         algos=ALGOS, wC=wC, wE=wE, C_ref=C_ref, E_ref=E_ref,
                         smoke=bool(args.smoke), dg_lim=BL.DG_LIM),
               runs=[{k: v for k, v in r.items() if k != "curve"} for r in recs],
               summary=summary)
    fn = f"comparison_dp{'_smoke' if args.smoke else ''}.json"
    with open(os.path.join(RESULTS, fn), "w", encoding="utf-8") as fp:
        json.dump(out, fp, ensure_ascii=False, indent=1)
    print(f"[完成] {fn}  总耗时 {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
