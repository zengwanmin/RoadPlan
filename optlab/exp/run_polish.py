# -*- coding: utf-8 -*-
"""
run_polish.py — v5: 在 v4 最优平面模态附近做收缩域温启动 IJS(+DP 内层)抛光。

外层搜索域收缩为 best_modes ± box(默认 0.06), 8 种子并行, 初始种群 = 最优点
+ 高斯扰动(σ=box/3), 逼近双层问题的局部最优。
"""
import os, json, time, argparse, multiprocessing as mp
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np

from data_loader import load_alignment
from objective_joint import make_plane_context, objectives_joint, N_MODE
from algorithms import run, VARIANTS
from run_bilevel import _full_x

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

_PC = None
_CTX = None


def _init(align, ctx):
    global _PC, _CTX
    _PC = make_plane_context(align)
    _CTX = ctx


def _solve(seed):
    pc, c = _PC, _CTX
    denom, lb, ub, m0 = c["denom"], c["lb"], c["ub"], c["m0"]

    def f(modes):
        xf = _full_x(pc, modes)
        C, E, pen, _ = objectives_joint(xf, pc, pen_scale=3.0)
        return (C + E) / denom + pen

    rng = np.random.default_rng(seed)
    sig = (ub - lb) / 6.0
    pop0 = np.clip(m0 + rng.normal(0, 1, (c["pop"], N_MODE)) * sig, lb, ub)
    pop0[0] = m0
    t0 = time.time()
    r = run(f, lb, ub, pop0, c["max_iter"], seed, **VARIANTS["V5_IJS"])
    xf = _full_x(pc, r["best_x"])
    C, E, pen, info = objectives_joint(xf, pc)
    return dict(seed=seed, C=float(C), E=float(E), CE=float(C + E),
                pen=float(pen), L_km=info["L_km"], Rmin=info["Rmin"],
                C_TU=info["C_TU"], L_bridge_new=info["L_bridge_new"],
                L_tunnel_new=info["L_tunnel_new"],
                wall_min=(time.time() - t0) / 60.0,
                best_x=xf.tolist(), modes=r["best_x"].tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="bilevel_v4.json")
    ap.add_argument("--tag", default="v5")
    ap.add_argument("--box", type=float, default=0.06)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--pop", type=int, default=32)
    ap.add_argument("--iter", type=int, default=120)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    t0 = time.time()
    with open(os.path.join(RESULTS, args.inp), encoding="utf-8") as f:
        res = json.load(f)
    denom = res["baseline"]["CE"]
    m0 = np.array(res["best"]["modes"], float)
    lb = np.clip(m0 - args.box, 0, 1)
    ub = np.clip(m0 + args.box, 0, 1)

    align = load_alignment()
    ctx = dict(denom=denom, m0=m0, lb=lb, ub=ub,
               pop=args.pop, max_iter=args.iter)
    print(f"[基线] C+E={denom/1e8:.4f}亿  起点降幅 "
          f"{(1-res['best']['CE']/denom)*100:.2f}%", flush=True)
    seeds = [7000 + 13 * k for k in range(args.seeds)]
    recs = []
    with mp.Pool(args.workers, initializer=_init, initargs=(align, ctx)) as pool:
        for r in pool.imap_unordered(_solve, seeds):
            recs.append(r)
            print(f"  seed={r['seed']} C+E={r['CE']/1e8:.4f}亿 "
                  f"({(r['CE']/denom-1)*100:+.2f}%) L={r['L_km']:.3f} "
                  f"Rmin={r['Rmin']:.0f} pen={r['pen']:.1e} "
                  f"[{r['wall_min']:.1f}min]", flush=True)
    feas = [r for r in recs if r["pen"] < 1e-9]
    best = min(feas or recs, key=lambda r: r["CE"])
    imp = (1 - best["CE"] / denom) * 100
    print(f"[最优] C+E={best['CE']/1e8:.4f}亿 降幅 {imp:.2f}% "
          f"(可行 {len(feas)}/{len(recs)})")
    out = dict(tag=args.tag, box=args.box, pop=args.pop, max_iter=args.iter,
               baseline=res["baseline"],
               runs=[{k: v for k, v in r.items() if k not in ("best_x", "modes")}
                     for r in recs],
               best=best, improvement_pct=imp,
               wall_min=(time.time() - t0) / 60.0)
    with open(os.path.join(RESULTS, f"polish_{args.tag}.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"[完成] polish_{args.tag}.json  总耗时 {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
