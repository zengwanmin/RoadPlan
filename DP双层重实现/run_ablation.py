# -*- coding: utf-8 -*-
"""
run_ablation.py — 消融实验(双层): IJS 各机制对【外层平面搜索】的贡献。

与旧版消融的本质差别:
  旧版是"固定平面 + 纵断面靠惩罚", 8 个变体的最优解 best_pen 全部落在
  5.9e8~1.5e9, 即全部违反纵坡/坡差规范 —— 上报的 C/E 不是可施工方案。
  本版纵断面由 DP 全局最优给出, 纵坡上下界与坡差由状态空间构造满足(pen 恒 0),
  故消融比较的是"外层平面搜索能力", 且每个变体的解都合规可施工。

【并行口径】默认按变体并行(每变体内部 n_runs 次串行):
  运行时间是上报列, 跨进程并行会受缓存/内存带宽争用干扰。按变体并行可使 8 个
  变体承受完全相同的并发条件, 故变体间运行时间仍可横向比较(绝对值随并发变化,
  须在表注中声明)。若要与单进程口径严格一致, 用 --serial。
  解的质量与并行无关: algorithms.run 只用 default_rng(seed) 局部生成器, 无全局
  RNG、无时间依赖, 故 best/mean/std/conv 在两种模式下逐位相同。
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
import bilevel as BL

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results"); os.makedirs(RESULTS, exist_ok=True)
_PC = None
_CTX = None


def convergence_gen(curve, frac=0.99):
    """达到最优解 99% 所需迭代次数。"""
    c = np.asarray(curve, float)
    f0, fs = c[0], c[-1]
    if abs(f0 - fs) < 1e-12:
        return 0
    below = np.where(c <= f0 - frac * (f0 - fs))[0]
    return int(below[0]) if len(below) else len(c) - 1


def _init(align, ctx, corridor, density_on):
    global _PC, _CTX
    OJ.set_corridor(corridor)
    OJ.set_density(density_on)
    _PC = make_plane_context(align)
    _CTX = ctx


def _run_variant(item):
    """一个变体的全部独立运行(内部串行, 保证同变体内时间口径一致)。"""
    vname, cfg = item
    pc, c = _PC, _CTX
    f = BL.make_outer_f(pc, c["wC"], c["wE"], c["C_ref"], c["E_ref"],
                        pen_scale=3.0)
    lb, ub = np.zeros(N_MODE), np.ones(N_MODE)
    best_fs, convs, rts, curves = [], [], [], []
    best_f, best_x = np.inf, None
    for r in range(c["n_runs"]):
        rng = np.random.default_rng(1000 + r)
        pop0 = rng.random((c["pop"], N_MODE))
        pop0[0] = 0.5                       # 现状平面必在初始种群内
        t0 = time.time()
        res = run(f, lb, ub, pop0, max_iter=c["max_iter"], seed=1000 + r, **cfg)
        rts.append(time.time() - t0)
        best_fs.append(float(res["best_f"]))
        convs.append(convergence_gen(res["curve"]))
        curves.append(np.asarray(res["curve"]).tolist())
        if res["best_f"] < best_f:
            best_f, best_x = res["best_f"], res["best_x"]
    ew = BL.dp_energy_weight(c["wC"], c["wE"], c["C_ref"], c["E_ref"])
    C, E, pen, info, z = BL.evaluate(best_x, pc, ew)
    d = BL.plane_terrain(best_x, pc)
    g = np.diff(z) / np.diff(d["sta_ctrl"])
    bf = np.array(best_fs)
    return dict(
        variant=vname, best=float(bf.min()), mean=float(bf.mean()),
        std=float(bf.std()), median=float(np.median(bf)),
        conv_gen_mean=float(np.mean(convs)), runtime_mean=float(np.mean(rts)),
        best_fs=bf.tolist(), conv_gens=list(map(int, convs)),
        runtimes=list(map(float, rts)),
        best_C=float(C), best_E=float(E), best_pen=float(pen),
        Rmin=float(info["Rmin"]), L_km=float(info["L_km"]),
        L_dense2_km=float(info["L_dense2_km"]),
        grade_max_pct=float(np.abs(g).max() * 100),
        grade_min_pct=float(np.abs(g).min() * 100),
        dgrade_max=float(np.abs(np.diff(g)).max()),
        curve_median=curves[int(np.argsort(bf)[len(bf) // 2])])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--corridor", type=float, default=1000.0)
    ap.add_argument("--pop", type=int, default=40)
    ap.add_argument("--iter", type=int, default=150)
    ap.add_argument("--n_runs", type=int, default=10)
    ap.add_argument("--serial", action="store_true",
                    help="单进程串行(运行时间口径与旧版严格一致, 但耗时数倍)")
    args = ap.parse_args()
    OJ.set_corridor(args.corridor)
    OJ.set_density(True)
    if args.smoke:
        args.pop, args.iter, args.n_runs = 8, 3, 2

    t0 = time.time()
    align = load_alignment()
    pc = make_plane_context(align)
    pop0, wC, wE, C_ref, E_ref = BL.baseline(pc, args.pop)
    print(f"[数据] 北环高速 {align['total_km']:.3f} km, 外层平面 {N_MODE} 维, "
          f"内层 DP 纵断面(全局最优)", flush=True)
    print(f"[熵权法] wC={wC:.4f}, wE={wE:.4f}", flush=True)
    print(f"[设置] {len(VARIANTS)} 变体 × {args.n_runs} 次, pop={args.pop}, "
          f"iter={args.iter}, {'串行' if args.serial else '按变体并行'}", flush=True)

    ctx = dict(wC=wC, wE=wE, C_ref=C_ref, E_ref=E_ref, pop=args.pop,
               max_iter=args.iter, n_runs=args.n_runs)
    items = list(VARIANTS.items())
    recs = []

    def _log(r):
        print(f"  [{r['variant']:16s}] best={r['best']:.5f} mean={r['mean']:.5f} "
              f"std={r['std']:.5f} conv={r['conv_gen_mean']:.1f} "
              f"t={r['runtime_mean']:.1f}s pen={r['best_pen']:.1e}", flush=True)

    if args.serial:
        _init(align, ctx, OJ.CORRIDOR_HALF_W, OJ.DENSITY_ON)
        for it in items:
            recs.append(_run_variant(it))
            _log(recs[-1])
    else:
        with mp.Pool(min(len(items), 8), initializer=_init,
                     initargs=(align, ctx, OJ.CORRIDOR_HALF_W,
                               OJ.DENSITY_ON)) as pool:
            for r in pool.imap_unordered(_run_variant, items):
                recs.append(r)
                _log(r)

    recs.sort(key=lambda r: r["mean"])
    bad = [r["variant"] for r in recs if r["best_pen"] > 1e-6]
    print(f"[合规] 违规变体: {bad if bad else '无(全部 pen=0)'}", flush=True)
    print("[排名] " + " < ".join(r["variant"] for r in recs), flush=True)

    out = dict(meta=dict(method="双层: 外层IJS变体消融 + 内层DP纵断面",
                         outer_dim=N_MODE, corridor_half_w=args.corridor,
                         pop=args.pop, max_iter=args.iter, n_runs=args.n_runs,
                         wC=wC, wE=wE, C_ref=C_ref, E_ref=E_ref,
                         execution=("串行单进程" if args.serial
                                    else "按变体并行(8进程)"),
                         smoke=bool(args.smoke), dg_lim=BL.DG_LIM),
               variants={r["variant"]: r for r in recs})
    fn = f"ablation_dp{'_smoke' if args.smoke else ''}.json"
    with open(os.path.join(RESULTS, fn), "w", encoding="utf-8") as fp:
        json.dump(out, fp, ensure_ascii=False, indent=1)
    print(f"[完成] {fn}  总耗时 {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
