# -*- coding: utf-8 -*-
"""
run_bilevel.py — v4 双层求解: 外层 IJS 搜平面模态(40维), 内层 DP 全局解纵断面。

依据:
  1. 双向平均口径下 E 对缓坡近似免费(实测同里程全平坡 E 比现状 +0.5%), 纵断面
     子问题 ≈ 坡度/坡差约束下的最小土方费 —— dp_profile.solve_profile 可 0.2 s
     全局求解(DZ=0.5 m 格), 坡度/坡差约束由构造满足。
  2. 于是联合问题分解为: 外层只搜 N_MODE 维平面(决定里程/曲率/沿线地形),
     每个平面用 DP 得到其最优纵断面, 再以完整目标 (C+E)/denom + pen 评价。
     265 维联合搜索 -> 40 维外层 + 精确内层。
"""
import os, json, time, argparse, multiprocessing as mp
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np

from data_loader import load_alignment
from objective_joint import (make_plane_context, objectives_joint, decode_joint,
                             DIM, N_MODE)
from algorithms import run, VARIANTS
from dp_profile import solve_profile, profile_to_x
import run_ce

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results"); os.makedirs(RESULTS, exist_ok=True)

_PC = None
_CTX = None


def _full_x(pc, modes_norm):
    """外层向量 -> 完整决策向量: 平面模态 + DP 最优纵断面。"""
    x = np.full(DIM, 0.5)
    x[:N_MODE] = np.clip(modes_norm, 0.0, 1.0)
    d = decode_joint(x, pc)                      # 仅用其平面/地形(纵断面占位)
    z_star, _ = solve_profile(d["sta_ctrl"], d["gz_ctrl"])
    return profile_to_x(z_star, d["gz_ctrl"], d["sta_ctrl"], x)


def _init(align, ctx):
    global _PC, _CTX
    _PC = make_plane_context(align)
    _CTX = ctx


def _solve(seed):
    pc, c = _PC, _CTX
    denom = c["denom"]

    def f(modes):
        xf = _full_x(pc, modes)
        C, E, pen, _ = objectives_joint(xf, pc, pen_scale=3.0)
        return (C + E) / denom + pen

    rng = np.random.default_rng(seed)
    pop0 = rng.random((c["pop"], N_MODE))
    pop0[0] = 0.5                                # 现状平面
    if c.get("chord_modes") is not None:
        pop0[1] = c["chord_modes"]
    t0 = time.time()
    r = run(f, np.zeros(N_MODE), np.ones(N_MODE), pop0, c["max_iter"], seed,
            **VARIANTS["V5_IJS"])
    xf = _full_x(pc, r["best_x"])
    C, E, pen, info = objectives_joint(xf, pc)
    return dict(seed=seed, C=float(C), E=float(E), CE=float(C + E),
                pen=float(pen), L_km=info["L_km"], Rmin=info["Rmin"],
                C_TU=info["C_TU"], CB=info["CB"], CS=info["CS"],
                L_bridge_new=info["L_bridge_new"],
                L_tunnel_new=info["L_tunnel_new"],
                wall_min=(time.time() - t0) / 60.0,
                best_x=xf.tolist(), modes=r["best_x"].tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v4")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--pop", type=int, default=32)
    ap.add_argument("--iter", type=int, default=150)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.iter, args.seeds, args.pop = 3, 2, 8

    t0 = time.time()
    align = load_alignment()
    pc = make_plane_context(align)
    x_A = run_ce.make_existing_x(pc)
    C_A, E_A, _, infoA = objectives_joint(x_A, pc)
    denom = C_A + E_A
    print(f"[基线 M-A] C+E={denom/1e8:.4f}亿", flush=True)
    x_ch, t_ch = run_ce.make_chord_x(pc, denom)
    ctx = dict(denom=denom, pop=args.pop, max_iter=args.iter,
               chord_modes=x_ch[:N_MODE])

    seeds = [5000 + 11 * k for k in range(args.seeds)]
    recs = []
    with mp.Pool(args.workers, initializer=_init, initargs=(align, ctx)) as pool:
        for r in pool.imap_unordered(_solve, seeds):
            recs.append(r)
            print(f"  seed={r['seed']} C+E={r['CE']/1e8:.4f}亿 "
                  f"({(r['CE']/denom-1)*100:+.2f}%) L={r['L_km']:.3f}km "
                  f"Rmin={r['Rmin']:.0f} C_TU={r['C_TU']/1e8:.3f} "
                  f"新桥={r['L_bridge_new']:.2f} 新隧={r['L_tunnel_new']:.2f} "
                  f"pen={r['pen']:.1e} [{r['wall_min']:.1f}min]", flush=True)
    feas = [r for r in recs if r["pen"] < 1e-9]
    best = min(feas or recs, key=lambda r: r["CE"])
    imp = (1 - best["CE"] / denom) * 100
    print(f"[最优] seed={best['seed']} C+E={best['CE']/1e8:.4f}亿 降幅 {imp:.2f}% "
          f"(可行 {len(feas)}/{len(recs)})")
    out = dict(tag=args.tag, pop=args.pop, max_iter=args.iter, seeds=seeds,
               baseline=dict(C=C_A, E=E_A, CE=denom),
               runs=[{k: v for k, v in r.items() if k not in ("best_x", "modes")}
                     for r in recs],
               best=best, improvement_pct=imp,
               wall_min=(time.time() - t0) / 60.0)
    fn = f"bilevel_{args.tag}{'_smoke' if args.smoke else ''}.json"
    with open(os.path.join(RESULTS, fn), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"[完成] {fn}  总耗时 {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
