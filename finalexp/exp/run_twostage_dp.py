# -*- coding: utf-8 -*-
"""
run_twostage_dp.py — 两阶段对照(定稿方法版): 先平面后纵断面, 与联合共用 DP 内层与熵权。

两阶段与联合的唯一差异 = 平面选线时是否预见纵断面代价:
  阶段一: IJS 只搜平面模态, 目标 = 仅随里程变化的平面 LCC(占地CR+基建CS+养护CQ 里程项)
          + 平曲线半径惩罚(论文 §3.4 的分阶段逻辑);
  阶段二: 冻结最优平面, DP 按熵权解纵断面(与联合的内层完全相同);
  联合  : 外层 IJS 的目标 = 完整 F(平面, DP最优纵断面)。(见 run_entropy_dp.py)
"""
import os, json, time, argparse
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np

from data_loader import load_alignment
from objective_joint import (make_plane_context, objectives_joint, decode_joint,
                             build_plane_from_delta, plane_lcc, make_scalar_plane,
                             DIM, N_MODE)
from algorithms import run, VARIANTS
from dp_profile import solve_profile, profile_to_x
import run_ce

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--pop", type=int, default=24)
    ap.add_argument("--iter", type=int, default=200)
    ap.add_argument("--fixw", required=True)
    ap.add_argument("--warm", default=None)
    args = ap.parse_args()
    wC, wE, C_ref, E_ref = (float(v) for v in args.fixw.split(","))

    t0 = time.time()
    align = load_alignment()
    pc = make_plane_context(align)
    x_A = run_ce.make_existing_x(pc)
    C_A, E_A, _, _ = objectives_joint(x_A, pc)
    ew = (wE / E_ref) / (wC / C_ref)

    # ---------- 阶段一: 平面(只看里程相关 LCC) ----------
    _, _, L0, _ = build_plane_from_delta(pc, np.full(N_MODE, 0.5))
    C_ref_plane = plane_lcc(L0)
    warm = None
    if args.warm and os.path.exists(os.path.join(RESULTS, args.warm)):
        warm = np.array(json.load(open(os.path.join(RESULTS, args.warm)))
                        ["best"]["modes"], float)[:N_MODE]
    best = None
    for k in range(args.seeds):
        seed = 9000 + 17 * k
        rng = np.random.default_rng(seed)
        pop0 = rng.random((args.pop, N_MODE))
        pop0[0] = 0.5
        if warm is not None:
            pop0[1] = warm
        fP = make_scalar_plane(pc, C_ref_plane, pen_scale=3.0)
        r = run(fP, np.zeros(N_MODE), np.ones(N_MODE), pop0, args.iter, seed,
                **VARIANTS["V5_IJS"])
        if best is None or r["best_f"] < best["best_f"]:
            best = r
    delta_star = best["best_x"]
    _, _, L_star, R_star = build_plane_from_delta(pc, delta_star)

    # ---------- 阶段二: 冻结平面, DP 解纵断面(熵权加权, 与联合同内层) ----------
    x = np.full(DIM, 0.5)
    x[:N_MODE] = delta_star
    d = decode_joint(x, pc)
    z_star, _ = solve_profile(d["sta_ctrl"], d["gz_ctrl"], energy_weight=ew)
    xf = profile_to_x(z_star, d["gz_ctrl"], d["sta_ctrl"], x)
    C, E, pen, info = objectives_joint(xf, pc)

    F = lambda c, e: wC * c / C_ref + wE * e / E_ref
    out = dict(tag=args.tag, method="twostage", wC=wC, wE=wE,
               C_ref=C_ref, E_ref=E_ref,
               baseline=dict(C=float(C_A), E=float(E_A), CE=float(C_A + E_A),
                             F=F(C_A, E_A)),
               best=dict(C=float(C), E=float(E), CE=float(C + E),
                         F=F(C, E), pen=float(pen), L_km=info["L_km"],
                         Rmin=info["Rmin"], C_TU=info["C_TU"],
                         L_bridge_new=info["L_bridge_new"],
                         L_tunnel_new=info["L_tunnel_new"],
                         net_drop_m=float(profile_end_drop(xf, pc)),
                         best_x=xf.tolist()),
               stage1=dict(L_km=L_star / 1000.0, Rmin=float(R_star.min())),
               improvement=dict(C=(1 - C / C_A) * 100, E=(1 - E / E_A) * 100,
                                CE=(1 - (C + E) / (C_A + E_A)) * 100,
                                F=(1 - F(C, E) / F(C_A, E_A)) * 100),
               wall_min=(time.time() - t0) / 60.0)
    print("[%s 两阶段] C %.2f%%  E %.2f%%  C+E %.2f%%  F %.2f%%  "
          "L=%.3f Rmin=%.0f pen=%.1e" % (
              args.tag, out["improvement"]["C"], out["improvement"]["E"],
              out["improvement"]["CE"], out["improvement"]["F"],
              info["L_km"], info["Rmin"], pen), flush=True)
    with open(os.path.join(RESULTS, "twostage_%s.json" % args.tag), "w",
              encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)


def profile_end_drop(xf, pc):
    d = decode_joint(np.asarray(xf, float), pc)
    return float(d["design_z"][-1] - d["design_z"][0])


if __name__ == "__main__":
    main()
