# -*- coding: utf-8 -*-
"""test_wsp.py — 单独验证 WSP(双暖启动+联合精修), 与 two_stage@500 配对比较(3 种子)。"""
import os, sys, time, json
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import multiprocessing as mp
from data_loader import load_alignment
from objective_joint import make_plane_context
import test_joint_improved as t


def _worker(seed_i):
    pc = make_plane_context(load_alignment())
    t0 = time.time()
    r = t.joint_wsp_once(pc, seed_i)
    r["seed_i"] = seed_i; r["wall_min"] = (time.time() - t0) / 60.0
    return r


if __name__ == "__main__":
    seeds = [int(s) for s in sys.argv[1:]] or [0, 1, 2]
    ts = {r["seed_i"]: r for r in
          json.load(open("results/budget_fairness_multiseed.json"))["raw"]["two_stage@500"]}
    t0 = time.time()
    res = {}
    with mp.Pool(min(len(seeds), 11)) as pool:
        for r in pool.imap_unordered(_worker, seeds):
            res[r["seed_i"]] = r
            print(f"  完成 seed{r['seed_i']} WSP C={r['C']/1e8:.4f} Rmin={r['Rmin']:.0f} "
                  f"pen={r['pen']:.1e} nev={r['nev']} ({r['wall_min']:.1f}min)", flush=True)
    print(f"\n{'seed':>4} {'method':>5} {'C(亿)':>9} {'E(亿)':>9} {'L(km)':>8} "
          f"{'Rmin':>6} {'pen':>9} {'nev':>8}  vs两阶段C")
    for si in seeds:
        b = ts[si]; r = res[si]
        print(f"{si:>4} {'2stg':>5} {b['C']/1e8:>9.4f} {b['E']/1e8:>9.4f} "
              f"{b['L_km']:>8.3f} {b['Rmin']:>6.0f} {b['pen']:>9.1e} {b['nev']:>8} {'baseline':>9}")
        dC = (r["C"] - b["C"]) / b["C"] * 100
        print(f"{si:>4} {'WSP':>5} {r['C']/1e8:>9.4f} {r['E']/1e8:>9.4f} "
              f"{r['L_km']:>8.3f} {r['Rmin']:>6.0f} {r['pen']:>9.1e} {r['nev']:>8} {dC:>+8.2f}%")
    print(f"\n总耗时 {(time.time()-t0)/60:.1f}min", flush=True)
