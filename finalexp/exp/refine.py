# -*- coding: utf-8 -*-
"""
refine.py — 对 run_ce 结果中的最优解做局部精修(L-BFGS-B, 有限差分梯度)。

依据: 目标 (C+E)/denom + pen 关于决策向量几乎处处连续可微(样条/双线性插值/min
封顶均连续), 单次求值仅毫秒级, 适合在 IJS 全局解附近做数值精修收尾。
惩罚用硬倍率(pen_scale=3), 精修后用默认倍率复核可行性。
"""
import os, json, time, argparse
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
from scipy.optimize import minimize

from data_loader import load_alignment
from objective_joint import make_plane_context, objectives_joint, DIM

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True, help="results/ce_*.json")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--maxiter", type=int, default=400)
    args = ap.parse_args()

    with open(os.path.join(RESULTS, args.inp) if not os.path.exists(args.inp)
              else args.inp, encoding="utf-8") as f:
        res = json.load(f)
    denom = res["baseline"]["CE"]
    x0 = np.array(res["best"]["best_x"], float)
    assert len(x0) == DIM

    align = load_alignment()
    pc = make_plane_context(align)

    neval = [0]

    def fun(x):
        neval[0] += 1
        C, E, pen, _ = objectives_joint(x, pc, pen_scale=3.0)
        return (C + E) / denom + pen

    t0 = time.time()
    f0 = fun(x0)
    r = minimize(fun, x0, method="L-BFGS-B",
                 bounds=[(0.0, 1.0)] * DIM,
                 options=dict(maxiter=args.maxiter, maxfun=400000, eps=1e-4))
    C, E, pen, info = objectives_joint(r.x, pc)
    C0, E0, pen0, _ = objectives_joint(x0, pc)
    print(f"[精修前] C+E={(C0+E0)/1e8:.4f}亿 ({(1-(C0+E0)/denom)*100:.2f}%) pen={pen0:.2e}")
    print(f"[精修后] C+E={(C+E)/1e8:.4f}亿 ({(1-(C+E)/denom)*100:.2f}%) pen={pen:.2e} "
          f"L={info['L_km']:.3f}km Rmin={info['Rmin']:.0f} "
          f"新桥={info['L_bridge_new']:.2f}km 新隧={info['L_tunnel_new']:.2f}km")
    print(f"  f {f0:.6f} -> {r.fun:.6f}  评估 {neval[0]} 次  "
          f"{(time.time()-t0)/60:.1f} min  ({r.message})")
    out = dict(tag=args.tag, from_json=args.inp, baseline=res["baseline"],
               C=float(C), E=float(E), CE=float(C + E), pen=float(pen),
               improvement_pct=(1 - (C + E) / denom) * 100,
               L_km=info["L_km"], Rmin=info["Rmin"],
               L_bridge_new=info["L_bridge_new"], L_tunnel_new=info["L_tunnel_new"],
               C_TU=info["C_TU"], CB=info["CB"], CS=info["CS"],
               best_x=r.x.tolist())
    with open(os.path.join(RESULTS, f"refine_{args.tag}.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"[完成] refine_{args.tag}.json")


if __name__ == "__main__":
    main()
