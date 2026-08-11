# -*- coding: utf-8 -*-
"""
run_entropy_dp.py — v9: 熵权法 + 双层求解(外层 IJS 平面模态, 内层加权 DP 纵断面)。

流程:
  1) joint_baseline 从基准种群客观算熵权 (wC, wE) 与参考尺度 (C_ref, E_ref);
  2) 内层 DP 转移费按熵权加权: energy_weight = (wE/E_ref)/(wC/C_ref);
  3) 外层 IJS 最小化 F = wC·C/C_ref + wE·E/E_ref + pen;
  4) 最优解做 L-BFGS-B 精修(同一 F)。
上报 F 与 C/E/C+E, 便于与直接 C+E 口径(v4/v5)对照。
"""
import os, json, time, argparse, multiprocessing as mp
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np

from data_loader import load_alignment
from objective_joint import (make_plane_context, objectives_joint, decode_joint,
                             joint_baseline, DIM, N_MODE)
from algorithms import run, VARIANTS
from dp_profile import solve_profile, profile_to_x
import run_ce

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results"); os.makedirs(RESULTS, exist_ok=True)

_PC = None
_CTX = None


def full_x_weighted(pc, modes_norm, ew):
    x = np.full(DIM, 0.5)
    x[:N_MODE] = np.clip(modes_norm, 0.0, 1.0)
    d = decode_joint(x, pc)
    z_star, _ = solve_profile(d["sta_ctrl"], d["gz_ctrl"], energy_weight=ew)
    return profile_to_x(z_star, d["gz_ctrl"], d["sta_ctrl"], x)


def _init(align, ctx):
    global _PC, _CTX
    _PC = make_plane_context(align)
    _CTX = ctx


def _F(C, E, c):
    return c["wC"] * C / c["C_ref"] + c["wE"] * E / c["E_ref"]


def _solve(seed):
    pc, c = _PC, _CTX

    def f(modes):
        xf = full_x_weighted(pc, modes, c["ew"])
        C, E, pen, _ = objectives_joint(xf, pc, pen_scale=3.0)
        return _F(C, E, c) + pen

    rng = np.random.default_rng(seed)
    pop0 = rng.random((c["pop"], N_MODE))
    pop0[0] = 0.5
    if c.get("warm_modes") is not None:
        pop0[1] = c["warm_modes"]
    t0 = time.time()
    r = run(f, np.zeros(N_MODE), np.ones(N_MODE), pop0, c["max_iter"], seed,
            **VARIANTS["V5_IJS"])
    xf = full_x_weighted(pc, r["best_x"], c["ew"])
    C, E, pen, info = objectives_joint(xf, pc)
    return dict(seed=seed, C=float(C), E=float(E), CE=float(C + E),
                F=float(_F(C, E, c)), pen=float(pen),
                L_km=info["L_km"], Rmin=info["Rmin"], C_TU=info["C_TU"],
                L_bridge_new=info["L_bridge_new"],
                L_tunnel_new=info["L_tunnel_new"],
                wall_min=(time.time() - t0) / 60.0,
                best_x=xf.tolist(), modes=r["best_x"].tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v9")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--pop", type=int, default=32)
    ap.add_argument("--iter", type=int, default=150)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--warm", default=None,
                    help="温启动 json(取其 best.modes; 维度不足补 0.5)")
    ap.add_argument("--fixw", default=None,
                    help="固定权重 wC,wE,C_ref,E_ref(跨场景可比)")
    args = ap.parse_args()
    if args.smoke:
        args.iter, args.seeds, args.pop = 3, 2, 8

    t0 = time.time()
    align = load_alignment()
    pc = make_plane_context(align)
    x_A = run_ce.make_existing_x(pc)
    C_A, E_A, _, infoA = objectives_joint(x_A, pc)

    if args.fixw:
        wC, wE, C_ref, E_ref = (float(v) for v in args.fixw.split(","))
    else:
        base, wC, wE, C_ref, E_ref = joint_baseline(pc, 200, x_seed=x_A)
    ew = (wE / E_ref) / (wC / C_ref)
    print(f"[熵权] wC={wC:.4f} wE={wE:.4f}  C_ref={C_ref/1e8:.2f}亿 "
          f"E_ref={E_ref/1e8:.2f}亿  DP能耗相对权重={ew:.3f}", flush=True)
    ctx = dict(wC=wC, wE=wE, C_ref=C_ref, E_ref=E_ref, ew=ew,
               pop=args.pop, max_iter=args.iter)
    F_A = _F(C_A, E_A, ctx)
    print(f"[基线 M-A] C={C_A/1e8:.4f}亿 E={E_A/1e8:.4f}亿 "
          f"C+E={(C_A+E_A)/1e8:.4f}亿 F={F_A:.6f}", flush=True)
    # 温启动: 用 warm_v8(重映射后的历史最优平面)
    wf = os.path.join(RESULTS, args.warm) if args.warm else         os.path.join(RESULTS, "warm_v8.json")
    if os.path.exists(wf):
        wm = np.array(json.load(open(wf))["best"]["modes"], float)
        if len(wm) < N_MODE:
            wm = np.concatenate([wm, np.full(N_MODE - len(wm), 0.5)])
        ctx["warm_modes"] = wm[:N_MODE]

    seeds = [9000 + 17 * k for k in range(args.seeds)]
    recs = []
    with mp.Pool(args.workers, initializer=_init, initargs=(align, ctx)) as pool:
        for r in pool.imap_unordered(_solve, seeds):
            recs.append(r)
            print(f"  seed={r['seed']} F={r['F']:.6f} "
                  f"C={r['C']/1e8:.4f} E={r['E']/1e8:.4f} "
                  f"C+E={r['CE']/1e8:.4f}亿({(r['CE']/(C_A+E_A)-1)*100:+.2f}%) "
                  f"L={r['L_km']:.3f} Rmin={r['Rmin']:.0f} pen={r['pen']:.1e} "
                  f"[{r['wall_min']:.1f}min]", flush=True)
    feas = [r for r in recs if r["pen"] < 1e-9]
    best = min(feas or recs, key=lambda r: r["F"])
    print(f"[最优F] seed={best['seed']} F={best['F']:.6f} "
          f"(基线 {F_A:.6f}, 降 {(1-best['F']/F_A)*100:.2f}%)  "
          f"C {(1-best['C']/C_A)*100:+.2f}%  E {(1-best['E']/E_A)*100:+.2f}%  "
          f"C+E {(1-best['CE']/(C_A+E_A))*100:+.2f}%")
    out = dict(tag=args.tag, wC=wC, wE=wE, C_ref=C_ref, E_ref=E_ref,
               ew=ew, pop=args.pop, max_iter=args.iter,
               baseline=dict(C=C_A, E=E_A, CE=C_A + E_A, F=F_A),
               runs=[{k: v for k, v in r.items() if k not in ("best_x", "modes")}
                     for r in recs],
               best=best,
               improvement_F_pct=(1 - best["F"] / F_A) * 100,
               improvement_CE_pct=(1 - best["CE"] / (C_A + E_A)) * 100,
               wall_min=(time.time() - t0) / 60.0)
    fn = f"entropy_dp_{args.tag}{'_smoke' if args.smoke else ''}.json"
    with open(os.path.join(RESULTS, fn), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"[完成] {fn}  总耗时 {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
