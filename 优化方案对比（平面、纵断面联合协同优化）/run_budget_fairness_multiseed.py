# -*- coding: utf-8 -*-
"""
run_budget_fairness_multiseed.py — 预算公平性核查的多次重复版(补充实验, 非官方四实验之一)

背景: 运行记录与问题定位.md §6.2 的预算公平性核查(两阶段 vs 联合按总求值量对齐)
每个预算点只跑了 1 次(单一种子), 结论(等预算下 C/E/L 打平但联合 Rmin 明显更高;
2×预算下两阶段反超但 Rmin 逼近约束边界)缺统计稳健性。本脚本对三个预算点各重复
N_RUNS=10 次独立随机种子, 报告 均值±标准差, 把单种子的观察转成可信的统计结论。

三个预算点(与 §6.2 定义一致):
  官方(不等预算): 两阶段(每阶段iter=500, 总求值~600,400) vs 联合(iter=500, ~300,200)
  等预算       : 两阶段(iter=500,          ~600,400) vs 联合(iter=1000, ~600,200)
  2×预算       : 两阶段(每阶段iter=1000,   ~1,200,400) vs 联合(iter=2000, ~1,200,200)

两阶段与联合的初始种群/算法种子均由 (方法, 预算点, 种子序号 i) 唯一确定,
与执行顺序/并行度无关(同项目其它实验的复现性约定)。
"""
import os, json, time, argparse, multiprocessing as mp
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np

from data_loader import load_alignment
from algorithms import run, VARIANTS
from objective import entropy_weights
from objective_joint import (N_CTRL, M_PROF, make_plane_context, objectives_joint,
                             make_scalar_joint, make_scalar_plane,
                             build_plane_from_delta, plane_lcc)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results"); os.makedirs(RESULTS, exist_ok=True)

POP = 200
SEED_BASE = 20250722
N_RUNS = 10


def two_stage_once(seed_i, iters):
    align = load_alignment(); pc = make_plane_context(align)
    rng = np.random.default_rng(SEED_BASE + seed_i)
    pop_plane = np.clip(0.5 + (rng.random((POP, N_CTRL)) - 0.5) * 1.0, 0, 1)
    _, _, L0, _ = build_plane_from_delta(pc, np.full(N_CTRL, 0.5))
    fP = make_scalar_plane(pc, plane_lcc(L0))
    rP = run(fP, np.zeros(N_CTRL), np.ones(N_CTRL), pop_plane, iters,
             SEED_BASE + 10000 + seed_i, **VARIANTS["V5_IJS"])
    ds = rP["best_x"]
    rng2 = np.random.default_rng(SEED_BASE + 20000 + seed_i)
    baseP = rng2.random((POP, M_PROF))
    full = lambda p: np.concatenate([ds, p])
    C0 = np.array([objectives_joint(full(baseP[k]), pc)[0] for k in range(POP)])
    E0 = np.array([objectives_joint(full(baseP[k]), pc)[1] for k in range(POP)])
    wC, wE = entropy_weights(C0, E0); C_ref, E_ref = float(C0.mean()), float(E0.mean())

    def sc(p):
        C, E, pen, _ = objectives_joint(full(p), pc)
        return wC * (C / C_ref) + wE * (E / E_ref) + pen / C_ref

    rC = run(sc, np.zeros(M_PROF), np.ones(M_PROF), baseP.copy(), iters,
             SEED_BASE + 30000 + seed_i, **VARIANTS["V5_IJS"])
    C, E, pen, info = objectives_joint(full(rC["best_x"]), pc)
    nev = 2 * (POP + 3 * POP * iters)
    return dict(C=float(C), E=float(E), L_km=float(info["L_km"]),
                Rmin=float(info["Rmin"]), pen=float(pen), nev=int(nev))


def joint_once(seed_i, iters):
    align = load_alignment(); pc = make_plane_context(align)
    dim = N_CTRL + M_PROF
    rng = np.random.default_rng(SEED_BASE + seed_i)
    base = np.empty((POP, dim))
    base[:, :N_CTRL] = 0.5 + (rng.random((POP, N_CTRL)) - 0.5) * 1.0
    base[:, N_CTRL:] = rng.random((POP, M_PROF))
    base = np.clip(base, 0, 1)
    C0 = np.array([objectives_joint(base[k], pc)[0] for k in range(POP)])
    E0 = np.array([objectives_joint(base[k], pc)[1] for k in range(POP)])
    wC, wE = entropy_weights(C0, E0); C_ref, E_ref = float(C0.mean()), float(E0.mean())
    f = make_scalar_joint(pc, wC, wE, C_ref, E_ref)
    r = run(f, np.zeros(dim), np.ones(dim), base, iters,
            SEED_BASE + 40000 + seed_i, **VARIANTS["V5_IJS"])
    C, E, pen, info = objectives_joint(r["best_x"], pc)
    nev = POP + 3 * POP * iters
    return dict(C=float(C), E=float(E), L_km=float(info["L_km"]),
                Rmin=float(info["Rmin"]), pen=float(pen), nev=int(nev))


def run_job(job):
    t0 = time.time()
    fn = two_stage_once if job["method"] == "two_stage" else joint_once
    rec = fn(job["seed_i"], job["iters"])
    rec.update(method=job["method"], iters=job["iters"], seed_i=job["seed_i"],
               wall_min=(time.time() - t0) / 60.0)
    return rec


def summarize(records):
    C = np.array([r["C"] for r in records]) / 1e8
    E = np.array([r["E"] for r in records]) / 1e8
    L = np.array([r["L_km"] for r in records])
    R = np.array([r["Rmin"] for r in records])
    P = np.array([r["pen"] for r in records])
    nev = records[0]["nev"]
    return dict(n=len(records), nev=nev,
                C_mean=float(C.mean()), C_std=float(C.std()),
                E_mean=float(E.mean()), E_std=float(E.std()),
                L_mean=float(L.mean()), L_std=float(L.std()),
                Rmin_mean=float(R.mean()), Rmin_std=float(R.std()),
                Rmin_min=float(R.min()), Rmin_max=float(R.max()),
                pen_mean=float(P.mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=11)
    ap.add_argument("--n_runs", type=int, default=N_RUNS)
    args = ap.parse_args()
    n_runs = args.n_runs

    jobs = []
    for i in range(n_runs):
        jobs.append(dict(method="two_stage", iters=500, seed_i=i, group="two_stage@500"))
        jobs.append(dict(method="two_stage", iters=1000, seed_i=i, group="two_stage@1000"))
        jobs.append(dict(method="joint", iters=500, seed_i=i, group="joint@500"))
        jobs.append(dict(method="joint", iters=1000, seed_i=i, group="joint@1000"))
        jobs.append(dict(method="joint", iters=2000, seed_i=i, group="joint@2000"))

    print(f"[启动] {len(jobs)} 个独立 IJS 寻优任务 (n_runs={n_runs} 种子 × 5 组), "
          f"{args.workers} 进程", flush=True)
    t0 = time.time()
    groups = {}
    with mp.Pool(args.workers) as pool:
        for k, rec in enumerate(pool.imap_unordered(run_job, jobs), 1):
            tag = f"{rec['method']}@{rec['iters']}"
            groups.setdefault(tag, []).append(rec)
            el = time.time() - t0
            print(f"  [{k:2d}/{len(jobs)}] {tag:14s} seed_i={rec['seed_i']} "
                  f"C={rec['C']/1e8:.4f}亿 Rmin={rec['Rmin']:.0f}m "
                  f"({rec['wall_min']:.1f}min/任务) | 累计{el/60:.1f}min "
                  f"ETA{el/k*(len(jobs)-k)/60:.1f}min", flush=True)

    summary = {tag: summarize(recs) for tag, recs in groups.items()}
    for tag, s in summary.items():
        print(f"[汇总] {tag:14s} n={s['n']} nev~{s['nev']} "
              f"C={s['C_mean']:.4f}±{s['C_std']:.4f}亿 "
              f"E={s['E_mean']:.4f}±{s['E_std']:.4f}亿 "
              f"L={s['L_mean']:.3f}±{s['L_std']:.3f}km "
              f"Rmin={s['Rmin_mean']:.0f}±{s['Rmin_std']:.0f}m "
              f"(范围{s['Rmin_min']:.0f}-{s['Rmin_max']:.0f}) pen={s['pen_mean']:.2e}")

    out = dict(n_runs=n_runs, seed_base=SEED_BASE,
               raw={tag: recs for tag, recs in groups.items()},
               summary=summary,
               total_wall_min=(time.time() - t0) / 60.0)
    fn = os.path.join(RESULTS, "budget_fairness_multiseed.json")
    with open(fn, "w", encoding="utf-8") as fp:
        json.dump(out, fp, ensure_ascii=False, indent=2)
    print(f"[完成] 总耗时 {(time.time()-t0)/60:.1f} min, 已写 {fn}")


if __name__ == "__main__":
    main()
