# -*- coding: utf-8 -*-
"""
run_sweep40_c8c.py — 图C8c: 仿论文图6.7 款式的 Pareto 解集(40 权重点, 本文口径)

协议与主实验 run_joint.py 完全一致(可引用):
  同一初始种群(joint_baseline, seed=2025, 现状解注入首位)、同一组熵权参考
  (wC/wE/C_ref/E_ref)、同一寻优管线(run_ijs_two_phase, pop=200, iter=1000,
  软/硬罚 0.3/3.0); 仅把权重扫描从 21 点加密到 40 点(w1 ∈ linspace(0,1,40))。

图C8c 款式仿论文图6.7: 红色星号散点、横轴能耗 E、纵轴全生命周期成本 C,
无辅助线; 图注注明本文口径(官方2025造价), 数值范围与论文图6.7 的差异
归因见《待办清单2》问题20(论文图轴单位矛盾+造价失真, 不做数字对齐)。

用法: python3 run_sweep40_c8c.py [--workers 30]
结果缓存 results/sweep40_c8c.json, 已存在则直接出图。
"""
import os, json, time, argparse, multiprocessing as mp
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES_FN = os.path.join(HERE, "results", "sweep40_c8c.json")
FIG = os.path.join(HERE, "figures")
C_YI = 1e8
N_W = 40

_CTX = None


def _init_worker():
    """worker 进程一次性构建 pc 与共享的 (base, wC, wE, C_ref, E_ref)。"""
    global _CTX
    from data_loader import load_alignment
    from objective_joint import make_plane_context, joint_baseline, DIM
    from run_joint import make_existing_x
    align = load_alignment()
    pc = make_plane_context(align)
    x_A = make_existing_x(pc, DIM)
    base, wC, wE, C_ref, E_ref = joint_baseline(pc, 200, x_seed=x_A)
    _CTX = dict(pc=pc, base=base, C_ref=C_ref, E_ref=E_ref)


def _one(job):
    """单个权重点: 与主实验相同的两阶段 IJS 寻优。"""
    i, w1 = job
    from objective_joint import (make_scalar_joint, run_ijs_two_phase,
                                 objectives_joint)
    pc = _CTX["pc"]; base = _CTX["base"]
    C_ref, E_ref = _CTX["C_ref"], _CTX["E_ref"]
    t0 = time.time()
    mk = lambda ps: make_scalar_joint(pc, w1, 1.0 - w1, C_ref, E_ref,
                                      pen_scale=ps)
    r = run_ijs_two_phase(mk, np.zeros(base.shape[1]), np.ones(base.shape[1]),
                          base.copy(), 1000, seed=3000 + i)
    C, E, pen, info = objectives_joint(r["best_x"], pc)
    print(f"  [{i+1:2d}/{N_W}] w1={w1:.3f} C={C/1e8:.2f}亿 E={E/1e8:.2f}亿 "
          f"pen={pen:.1e} | {(time.time()-t0)/60:.0f}min", flush=True)
    return dict(w1=float(w1), C=float(C), E=float(E), pen=float(pen),
                L_km=float(info["L_km"]))


def run_sweep(workers):
    w1s = np.linspace(0.0, 1.0, N_W)
    jobs = list(enumerate(w1s))
    t0 = time.time()
    with mp.Pool(min(workers, N_W), initializer=_init_worker) as pool:
        recs = pool.map(_one, jobs)
    out = dict(meta=dict(n_weights=N_W, pop=200, max_iter=1000,
                         protocol="run_ijs_two_phase soft0.3/hard3.0, "
                                  "joint_baseline seed2025+现状注入, 交叉桥内生口径",
                         minutes=round((time.time() - t0) / 60, 1)),
               sweep=recs)
    with open(RES_FN, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[完成] {RES_FN}  耗时 {(time.time()-t0)/60:.0f} min")
    return out


def plot(d):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["font.size"] = 11
    sw = d["sweep"]
    feas = [p for p in sw if p["pen"] <= 1e-6]
    E = np.array([p["E"] for p in feas]) / C_YI
    C = np.array([p["C"] for p in feas]) / C_YI
    plt.figure(figsize=(7.4, 5.4))
    plt.plot(E, C, "*", color="red", ms=10, mew=1.2, ls="none")
    plt.xlabel("Energy consumption E (10^8 RMB)")
    plt.ylabel("Life-cycle cost C (10^8 RMB)")
    plt.title("Fig. C8c  Pareto solution set (40 weight points, style of thesis Fig. 6.7)")
    plt.annotate("Cost basis: official Guangzhou 2025 unit prices;\n"
                 "endogenous crossing bridges & eco-tunnel.\n"
                 "Axis ranges differ from thesis Fig. 6.7 by design\n"
                 "(unit inconsistency there; see checklist issue #20).",
                 xy=(0.97, 0.97), xycoords="axes fraction", ha="right",
                 va="top", fontsize=8.5, color="#555555")
    plt.grid(alpha=0.3)
    for ext in ("png", "pdf"):
        plt.savefig(os.path.join(
            FIG, f"图C8c_仿图6.7款式Pareto解集_40点.{ext}"),
            bbox_inches="tight")
    print(f"[图] figures/图C8c_仿图6.7款式Pareto解集_40点  "
          f"(可行 {len(feas)}/{len(sw)} 点)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=30)
    a = ap.parse_args()
    if os.path.exists(RES_FN):
        with open(RES_FN, encoding="utf-8") as f:
            d = json.load(f)
        print("[缓存] 使用已有 sweep40_c8c.json")
    else:
        d = run_sweep(a.workers)
    plot(d)
