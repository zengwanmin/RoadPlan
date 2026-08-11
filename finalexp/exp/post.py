# -*- coding: utf-8 -*-
"""post.py — 批量 L-BFGS-B 精修(硬惩罚) + 单向解的双向交叉重评, 输出 final_table.json。"""
import os, sys, json, glob
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")


def refine_one(task):
    tag, kind, W, mode = task
    os.environ["CORRIDOR_HALF_W"] = str(W)
    os.environ["E_DIRECTION"] = mode
    for m in list(sys.modules):
        if m in ("objective_joint", "objective", "dp_profile", "run_ce",
                 "data_loader", "dem"):
            del sys.modules[m]
    from data_loader import load_alignment
    import objective_joint as oj
    import run_ce

    fp = os.path.join(RESULTS, ("entropy_dp_%s.json" if kind == "joint"
                                else "twostage_%s.json") % tag)
    d = json.load(open(fp))
    wC, wE, Cr, Er = d["wC"], d["wE"], d["C_ref"], d["E_ref"]
    x0 = np.array(d["best"]["best_x"], float)
    pc = oj.make_plane_context(load_alignment())

    def fun(x):
        C, E, pen, _ = oj.objectives_joint(x, pc, pen_scale=10.0)
        return wC * C / Cr + wE * E / Er + pen

    r = minimize(fun, x0, method="L-BFGS-B", bounds=[(0, 1)] * len(x0),
                 options=dict(maxiter=300, maxfun=300000, eps=1e-4))
    C, E, pen, info = oj.objectives_joint(r.x, pc)
    x_A = run_ce.make_existing_x(pc)
    C_A, E_A, _, _ = oj.objectives_joint(x_A, pc)
    rec = dict(tag=tag, kind=kind, W=W, mode=mode,
               C=float(C), E=float(E), CE=float(C + E), pen=float(pen),
               F=wC * C / Cr + wE * E / Er,
               F_A=wC * C_A / Cr + wE * E_A / Er,
               C_A=float(C_A), E_A=float(E_A),
               impC=(1 - C / C_A) * 100, impE=(1 - E / E_A) * 100,
               impCE=(1 - (C + E) / (C_A + E_A)) * 100,
               L_km=info["L_km"], Rmin=info["Rmin"], C_TU=info["C_TU"],
               L_bridge_new=info["L_bridge_new"],
               net_drop_m=float(_net_drop(oj, r.x, pc)))
    # 单向解 -> 双向口径交叉重评
    if mode == "single":
        os.environ["E_DIRECTION"] = "avg"
        for m in ("objective_joint", "objective", "dp_profile", "run_ce"):
            if m in sys.modules:
                del sys.modules[m]
        import objective_joint as oj2
        import run_ce as rc2
        pc2 = oj2.make_plane_context(load_alignment())
        C2, E2, _, _ = oj2.objectives_joint(r.x, pc2)
        xA2 = rc2.make_existing_x(pc2)
        CA2, EA2, _, _ = oj2.objectives_joint(xA2, pc2)
        rec["impE_crosscheck_avg"] = (1 - E2 / EA2) * 100
    np.save(os.path.join(RESULTS, "xref_%s_%s.npy" % (kind, tag)), r.x)
    return rec


def main():
    tasks = []
    for W in (500, 600, 700, 800, 900, 1000):
        for mode in ("avg", "single"):
            tasks.append(("%s_w%d" % (mode, W), "joint", W, mode))
    for mode in ("avg", "single"):
        tasks.append(("%s_w500" % mode, "twostage", 500, mode))
    import multiprocessing as mp
    with mp.Pool(14) as pool:
        recs = pool.map(refine_one, tasks)
    with open(os.path.join(RESULTS, "final_table.json"), "w",
              encoding="utf-8") as f:
        json.dump(recs, f, ensure_ascii=False, indent=1)
    for r in recs:
        cx = r.get("impE_crosscheck_avg")
        print("%-9s %-16s C %6.2f%%  E %6.2f%%  C+E %6.2f%%  pen %.1e  "
              "L=%.3f Rmin=%.0f 净落差 %+6.1f m%s" % (
                  r["kind"], r["tag"], r["impC"], r["impE"], r["impCE"],
                  r["pen"], r["L_km"], r["Rmin"], r["net_drop_m"],
                  ("  [双向重评E %.2f%%]" % cx) if cx is not None else ""))


def _net_drop(oj, x, pc):
    d = oj.decode_joint(np.asarray(x, float), pc)
    return d["design_z"][-1] - d["design_z"][0]


if __name__ == "__main__":
    main()
