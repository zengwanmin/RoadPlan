# -*- coding: utf-8 -*-
"""rerun_corridor.py — 只重跑项目⑧(走廊带敏感性, 6点)并合并进 reopt_results.json。

首轮全量运行时 item8 存在两个问题(已修复于 run_reopt.py):
  1) 逐点熵权在宽走廊带下落入"全线高架"退化区 -> 改为固定 w1=0.65(主实验决策点);
  2) _assemble 未保存 item8 -> 已补 corridor 键。
本脚本按修复后的口径重算 6 个走廊带点(种子与全量运行的任务序一致), 合并保存。
"""
import json
import multiprocessing as mp
import os
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np

import run_reopt as R
from data_loader import load_alignment
from params import TRAFFIC

HERE = os.path.dirname(os.path.abspath(__file__))
FN = os.path.join(HERE, "results", "reopt_results.json")


def main():
    t0 = time.time()
    align = load_alignment()
    # 与 run_reopt.main 正式网格完全一致(保证 item8 任务种子不变)
    grids = dict(
        traffic=list(np.array([0, 2, 4, 6, 8, 10]) / 100.0),
        ev=list(np.linspace(0, 1.0, 21)),
        fuel_price=list(np.linspace(0, 0.05, 6)),
        elec_price=list(np.linspace(0, 0.05, 6)),
        fuel_save=list(np.linspace(0, 0.05, 6)),
        elec_save=list(np.linspace(0, 0.05, 6)),
        w1=list(np.linspace(0.1, 0.9, 9)),
        w1_cost=list(np.linspace(0.7, 1.0, 7)),
        w1_energy=list(np.linspace(0.0, 0.2, 5)),
        w1_balanced=list(np.linspace(0.3, 0.7, 9)),
        corridor=[200, 250, 500, 1000, 2000, 2500],
    )
    tasks = [t for t in R.build_tasks(grids) if t["item"] == 8]
    print(f"[任务] 走廊带 {len(tasks)} 点, w1=0.65 固定权重, "
          f"pop{R.POP_SIZE}/iter{R.MAX_ITER}")
    recs = []
    with mp.Pool(len(tasks), initializer=R._init_worker,
                 initargs=(align, R.POP_SIZE, R.MAX_ITER)) as pool:
        for k, rec in enumerate(pool.imap_unordered(R._optimize_one, tasks), 1):
            recs.append(rec)
            print(f"  [{k}/{len(tasks)}] ±{rec['corridor']:.0f}m "
                  f"C={rec['C']/1e8:.4f}亿 E={rec['E']/1e8:.4f}亿 "
                  f"L={rec['L_km']:.3f}km 隧道={rec['L_eco_km']:.2f}km "
                  f"pen={rec['pen']:.1e}", flush=True)

    corridor = [dict(corridor=r["corridor"], C=r["C"], E=r["E"],
                     L_km=r["L_km"], Rmin=r["Rmin"], pen=r["pen"],
                     L_eco_km=r["L_eco_km"], wC=r["wC"])
                for r in sorted(recs, key=lambda r: r["corridor"])]
    with open(FN, encoding="utf-8") as f:
        out = json.load(f)
    out["corridor"] = corridor
    out["meta"]["corridor_note"] = ("item8 走廊带敏感性: 固定 w1=0.65"
                                    "(主实验前沿熵权决策点), 修复口径后单独重算合并")
    with open(FN, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[完成] corridor 键已合并入 {FN}  耗时 {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
