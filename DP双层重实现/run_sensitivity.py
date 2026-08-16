# -*- coding: utf-8 -*-
"""
run_sensitivity.py — 敏感性分析(双层): 每个采样点都做一次完整重优化。

敏感性项:
  item1 traffic       交通量年增长率 rj
  item2 ev            电动车渗透率
  item3 fuel_price    油价年增长率      item4 elec_price 电价年增长率
  item5 fuel_save     节油率            item6 elec_save  节能率
  item7 w1            成本权重 wC(能耗权重 1-wC)
  item8 corridor      走廊带半宽(m)     —— 本项的自变量就是走廊带, 故不固定
  item9 dp_dz         DP 高程格(m)      —— 双层法特有: 内层离散精度的敏感性
  item10 e_direction  能耗方向口径(单向/双向平均)

【走廊带的处理】除 item8 外, 所有项的基准走廊带固定为 --corridor(默认 1000 m);
item8 本身扫 [200…2500], 否则该实验项将不存在。

【dp_dz 项的意义】DP 把高程离散成 DZ 的格子, DZ 越大解越粗、求解越快。该项回答
"内层离散精度是否影响结论", 是双层法必须自证的一条(格点过粗会让纵坡台阶变粗)。
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
SEED_BASE = 20250722
_ALIGN = None
_CTX = None


def _init(align, ctx):
    global _ALIGN, _CTX
    _ALIGN, _CTX = align, ctx


def _solve(task):
    c = _CTX
    # 走廊带与 DP 格距须在建上下文/求解前设置
    OJ.set_corridor(task.get("corridor", c["corridor"]))
    OJ.set_density(True)
    import dp_profile
    dp_profile.DP_DZ = float(task.get("dp_dz", c["dp_dz"]))
    if "e_direction" in task:
        os.environ["E_DIRECTION"] = task["e_direction"]
    pc = make_plane_context(_ALIGN)

    wC = float(task.get("w1", c["wC"]))
    wE = 1.0 - wC
    scen = {k: v for k, v in task.items()
            if k in ("ev", "traffic_growth", "fuel_price_growth",
                     "elec_price_growth", "fuel_save", "elec_save")}
    f = BL.make_outer_f(pc, wC, wE, c["C_ref"], c["E_ref"], pen_scale=3.0,
                        scenario=scen or None)
    rng = np.random.default_rng(task["seed"])
    pop0 = rng.random((c["pop"], N_MODE))
    pop0[0] = 0.5
    t0 = time.time()
    r = run(f, np.zeros(N_MODE), np.ones(N_MODE), pop0, c["iter"],
            task["seed"], **VARIANTS["V5_IJS"])
    ew = BL.dp_energy_weight(wC, wE, c["C_ref"], c["E_ref"])
    C, E, pen, info, z = BL.evaluate(r["best_x"], pc, ew, scenario=scen or None)
    CA, EA, _, _, _ = BL.evaluate_existing(pc, scenario=scen or None)
    d = BL.plane_terrain(r["best_x"], pc)
    g = np.diff(z) / np.diff(d["sta_ctrl"])
    return dict(item=task["item"], label=task["label"], value=task["value"],
                C=float(C), E=float(E), pen=float(pen),
                C_A=float(CA), E_A=float(EA),
                dC_pct=float((C / CA - 1) * 100),
                dE_pct=float((E / EA - 1) * 100),
                Rmin=float(info["Rmin"]), L_km=float(info["L_km"]),
                L_dense2_km=float(info["L_dense2_km"]),
                grade_max_pct=float(np.abs(g).max() * 100),
                grade_min_pct=float(np.abs(g).min() * 100),
                dgrade_max=float(np.abs(np.diff(g)).max()),
                wall_s=time.time() - t0)


def build_tasks(smoke, corridor):
    """构造采样点。每点一个固定种子(SEED_BASE+序号), 可复现。"""
    if smoke:
        G = dict(traffic=[0.0, 0.1], ev=[0.0, 1.0], fuel_price=[0.05],
                 elec_price=[0.05], fuel_save=[0.05], elec_save=[0.05],
                 w1=[0.1, 0.9], corridor=[250, 2500], dp_dz=[0.5, 1.0],
                 e_direction=["single", "avg"])
    else:
        G = dict(traffic=list(np.linspace(0, 0.10, 6)),
                 ev=list(np.linspace(0, 1.0, 6)),
                 fuel_price=list(np.linspace(0, 0.05, 6)),
                 elec_price=list(np.linspace(0, 0.05, 6)),
                 fuel_save=list(np.linspace(0, 0.05, 6)),
                 elec_save=list(np.linspace(0, 0.05, 6)),
                 w1=list(np.linspace(0.1, 0.9, 9)),
                 corridor=[200, 250, 500, 1000, 2000, 2500],
                 dp_dz=[0.25, 0.5, 1.0, 2.0],
                 e_direction=["single", "avg"])
    key = dict(traffic="traffic_growth", ev="ev",
               fuel_price="fuel_price_growth", elec_price="elec_price_growth",
               fuel_save="fuel_save", elec_save="elec_save",
               w1="w1", corridor="corridor", dp_dz="dp_dz",
               e_direction="e_direction")
    tasks, gid = [], 0
    for i, (name, vals) in enumerate(G.items(), 1):
        for v in vals:
            t = dict(item=i, label=name, value=v, seed=SEED_BASE + gid)
            t[key[name]] = v
            tasks.append(t)
            gid += 1
    return tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--corridor", type=float, default=1000.0,
                    help="除 item8 外各项的基准走廊带半宽(m)")
    ap.add_argument("--pop", type=int, default=40)
    ap.add_argument("--iter", type=int, default=150)
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()
    if args.smoke:
        args.pop, args.iter = 8, 3

    t0 = time.time()
    align = load_alignment()
    OJ.set_corridor(args.corridor)
    OJ.set_density(True)
    pc = make_plane_context(align)
    pop0, wC, wE, C_ref, E_ref = BL.baseline(pc, args.pop)
    print(f"[数据] 北环高速 {align['total_km']:.3f} km, 外层平面 {N_MODE} 维, "
          f"内层 DP 纵断面(全局最优)", flush=True)
    print(f"[熵权法] wC={wC:.4f}, wE={wE:.4f} (基准口径, w1 项会覆盖)", flush=True)

    tasks = build_tasks(args.smoke, args.corridor)
    ctx = dict(wC=wC, wE=wE, C_ref=C_ref, E_ref=E_ref, pop=args.pop,
               iter=args.iter, corridor=args.corridor, dp_dz=0.5)
    nw = args.workers or min(len(tasks), max(1, (os.cpu_count() or 2) - 2))
    print(f"[任务] {len(tasks)} 个采样点, 各重优化一次, {nw} 进程", flush=True)

    recs = []
    with mp.Pool(nw, initializer=_init, initargs=(align, ctx)) as pool:
        for k, r in enumerate(pool.imap_unordered(_solve, tasks), 1):
            recs.append(r)
            print(f"  [{k:3d}/{len(tasks)}] item{r['item']} {r['label']:12s}"
                  f"={r['value']!s:>8} ΔC={r['dC_pct']:+6.2f}% "
                  f"ΔE={r['dE_pct']:+6.2f}% Rmin={r['Rmin']:.0f} "
                  f"pen={r['pen']:.1e} [{r['wall_s']:.1f}s]", flush=True)

    # 各项极差(反映敏感度), 并做合规复核
    print("\n[敏感度] 各项 ΔC/ΔE 极差:", flush=True)
    tornado = {}
    for name in sorted({r["label"] for r in recs}):
        sub = [r for r in recs if r["label"] == name]
        dc = [r["dC_pct"] for r in sub]; de = [r["dE_pct"] for r in sub]
        tornado[name] = dict(C_range=float(max(dc) - min(dc)),
                             E_range=float(max(de) - min(de)), n=len(sub))
        print(f"  {name:12s} C极差={tornado[name]['C_range']:6.2f}pp "
              f"E极差={tornado[name]['E_range']:6.2f}pp", flush=True)
    bad = [r for r in recs if r["grade_max_pct"] > 4.0 + 1e-6
           or r["grade_min_pct"] < 0.3 - 1e-6
           or r["dgrade_max"] > BL.DG_LIM + 1e-9 or r["L_dense2_km"] > 1e-9]
    print(f"[合规] 违规采样点: {len(bad)}/{len(recs)}", flush=True)

    out = dict(meta=dict(method="双层: 外层IJS平面 + 内层DP纵断面, 逐点重优化",
                         outer_dim=N_MODE, base_corridor=args.corridor,
                         pop=args.pop, max_iter=args.iter, seed_base=SEED_BASE,
                         wC=wC, wE=wE, C_ref=C_ref, E_ref=E_ref,
                         smoke=bool(args.smoke), dg_lim=BL.DG_LIM),
               runs=recs, tornado=tornado,
               n_violation=len(bad))
    fn = f"sensitivity_dp{'_smoke' if args.smoke else ''}.json"
    with open(os.path.join(RESULTS, fn), "w", encoding="utf-8") as fp:
        json.dump(out, fp, ensure_ascii=False, indent=1)
    print(f"[完成] {fn}  总耗时 {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
