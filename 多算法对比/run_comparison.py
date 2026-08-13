# -*- coding: utf-8 -*-
"""
run_comparison.py — 多算法对比主程序 (实验设计方案2 · 实验二, 2026-08-12 升级)

【PJ1-PJ6 平纵联合规模阶梯】(待办清单2 问题19, 替代旧版 P1-P6 纯纵断面口径)
  在与主实验完全相同的【平纵联合】模型(objective_joint: 平面50正弦模态+纵断面
  变坡点, 准天然地面DEM, 交叉桥内生触发+生态隧道内生, ±500m 走廊带)上, 以
  【纵断面变坡点步长】生成规模阶梯(平面模态数固定50, 避免双变量混淆):

    PJ1 500m(dim≈95)  PJ2 400m(≈106)  PJ3 300m(≈125)
    PJ4 200m(≈162)    PJ5 100m(=275, 主实验口径)  PJ6 50m(≈499)

  对比算法: IJS(本文) / JS(原型) / NSGA-II(学位论文原算法) / GA / PSO / GWO,
  每算法每规模 10 次独立运行(联合求值成本高, 减种子数并声明), pop=200, iter=500
  (NFE 与主实验对齐; IJS 因 Levy/DE 阶段每代 3×NFE, 见 NFE_MULT 列声明)。
  惩罚采用单一 pen_scale=1.0(所有算法同一目标函数, 保证公平; 不用主实验的
  两阶段软硬调度, 因其为 IJS 专用管线)。

【并行粒度: 按 (规模, 算法) 并行, 单元内部串行】
  36 个 (规模,算法) 单元各占一个单核进程(BLAS 单线程), 单元内 10 次独立运行
  串行执行; 运行时间在同构单核条件下测得, 算法间耗时可比(声明测时口径)。
  每次运行的初始种群与种子由 (run 序号) 唯一确定, 与执行顺序无关。

用法:
  python3 run_comparison.py                 # 正式全量 (6规模×6算法×10次)
  python3 run_comparison.py --smoke         # 冒烟 (iter=5, 2次, PJ1/PJ6)
  python3 run_comparison.py --workers 30    # 限制并发进程数(CPU 预算)
"""
import os, json, time, argparse, multiprocessing as mp
# 多进程下禁用 BLAS 内部多线程(每进程 1 核), 否则抢核使 runtime 失真。
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np

from params import ALGO
from data_loader import load_alignment
from algorithms import run, VARIANTS
from benchmarks import run_GA, run_PSO, run_GWO, run_NSGA2
from metrics import wilcoxon_ranksum, friedman_ranks

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)

# 六组联合规模: 纵断面变坡点步长 500/400/300/200/100/50 m (问题19 定案)
SCALES = {
    "PJ1": dict(step_m=500.0, label="PJ1 (profile step 500 m)"),
    "PJ2": dict(step_m=400.0, label="PJ2 (profile step 400 m)"),
    "PJ3": dict(step_m=300.0, label="PJ3 (profile step 300 m)"),
    "PJ4": dict(step_m=200.0, label="PJ4 (profile step 200 m)"),
    "PJ5": dict(step_m=100.0, label="PJ5 (profile step 100 m, main-exp)"),
    "PJ6": dict(step_m=50.0,  label="PJ6 (profile step 50 m)"),
}
ALGOS = ["IJS", "JS", "NSGA-II", "GA", "PSO", "GWO"]
N_RUNS_PJ = 10       # 联合求值成本高, 每单元 10 种子(声明)

_PC = None           # worker 内缓存的平面上下文


def _ctx(step_m):
    """worker 内构建(缓存)联合上下文: 切换步长 + 平面上下文。"""
    global _PC
    import objective_joint as OJ
    OJ.set_profile_step(step_m)
    if _PC is None:
        _PC = OJ.make_plane_context(load_alignment())
    return OJ, _PC


def make_scalar_joint_fn(step_m, wC, wE, C_ref, E_ref):
    OJ, pc = _ctx(step_m)
    def f(x):
        C, E, pen, _ = OJ.objectives_joint(x, pc, pen_scale=1.0)
        return wC * (C / C_ref) + wE * (E / E_ref) + pen
    return f


def make_biobj_joint_fn(step_m, C_ref, E_ref):
    OJ, pc = _ctx(step_m)
    def f(x):
        C, E, pen, _ = OJ.objectives_joint(x, pc, pen_scale=1.0)
        return np.array([C / C_ref + pen, E / E_ref + pen])
    return f


def convergence_gen(curve, frac=0.99):
    f0, fstar = curve[0], curve[-1]
    if abs(f0 - fstar) < 1e-12:
        return 0
    target = f0 - frac * (f0 - fstar)
    below = np.where(curve <= target)[0]
    return int(below[0]) if len(below) else len(curve) - 1


def run_one_scalar(algo, f, lb, ub, pop0, max_iter, seed):
    t0 = time.time()
    if algo == "IJS":
        r = run(f, lb, ub, pop0, max_iter, seed, **VARIANTS["V5_IJS"])
    elif algo == "JS":
        r = run(f, lb, ub, pop0, max_iter, seed, **VARIANTS["V1_JS"])
    elif algo == "GA":
        r = run_GA(f, lb, ub, pop0, max_iter, seed, p1=0.8, p2=0.2)
    elif algo == "PSO":
        r = run_PSO(f, lb, ub, pop0, max_iter, seed, w=0.8, c1=2.0, c2=2.0)
    elif algo == "GWO":
        r = run_GWO(f, lb, ub, pop0, max_iter, seed)
    else:
        raise ValueError(algo)
    return r["best_f"], r["curve"], time.time() - t0, r["best_x"]


def run_unit(job):
    """
    单个 (规模, 算法) 单元: n_runs 次独立运行串行执行(单核进程, 耗时可比)。
    返回 (sk, algo, unit_res)。
    """
    (sk, algo, step_m, wC, wE, C_ref, E_ref,
     pop_size, max_iter, n_runs) = job
    OJ, pc = _ctx(step_m)
    dim = OJ.DIM
    lb, ub = np.zeros(dim), np.ones(dim)
    best_fs, conv_gens, runtimes = [], [], []
    curves, feas = [], []
    for r in range(n_runs):
        rng = np.random.default_rng(1000 + r)
        pop0 = rng.random((pop_size, dim))
        if algo == "NSGA-II":
            fbi = make_biobj_joint_fn(step_m, C_ref, E_ref)
            t0 = time.time()
            rn = run_NSGA2(fbi, lb, ub, pop0, max_iter, 1000 + r)
            rt = time.time() - t0
            fr = rn["front_F"]
            scal = wC * fr[:, 0] + wE * fr[:, 1]
            bi = int(np.argmin(scal))
            bf = float(scal[bi])
            bx = rn["front_X"][bi] if "front_X" in rn else None
            best_fs.append(bf); conv_gens.append(max_iter); runtimes.append(rt)
            curves.append(None)
        else:
            f = make_scalar_joint_fn(step_m, wC, wE, C_ref, E_ref)
            bf, curve, rt, bx = run_one_scalar(algo, f, lb, ub, pop0,
                                               max_iter, 1000 + r)
            best_fs.append(bf)
            conv_gens.append(convergence_gen(curve))
            runtimes.append(rt)
            curves.append(curve)
        # 可行性(pen==0)与进入可行域代数(问题19 加分项: 两段收敛报告)
        if bx is not None:
            C, E, pen, info = OJ.objectives_joint(np.asarray(bx), pc)
            feas.append(dict(penalty=float(pen), C=float(C), E=float(E),
                             L_km=float(info["L_km"]),
                             L_cross_km=float(info["L_cross_km"])))
        else:
            feas.append(None)
    best_fs = np.array(best_fs)
    med_idx = int(np.argsort(best_fs)[len(best_fs) // 2])
    med_curve = curves[med_idx]
    unit = dict(
        best=float(best_fs.min()), mean=float(best_fs.mean()),
        std=float(best_fs.std()), median=float(np.median(best_fs)),
        conv_gen_mean=float(np.mean(conv_gens)),
        runtime_mean=float(np.mean(runtimes)),
        best_fs=best_fs.tolist(),
        runtimes=list(map(float, runtimes)),
        feas=feas,
        curve=(None if med_curve is None else np.asarray(med_curve).tolist()))
    print(f"  [{sk}][{algo:8s}] best={best_fs.min():.4f} "
          f"mean={best_fs.mean():.4f} std={best_fs.std():.4f} "
          f"t={np.mean(runtimes):.1f}s", flush=True)
    return sk, algo, unit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="冒烟(iter=5, n_runs=2, 仅 PJ1/PJ6)")
    ap.add_argument("--workers", type=int, default=30,
                    help="最大并发进程数(默认 30, CPU 预算 32 核)")
    args = ap.parse_args()

    t_all = time.time()
    import objective_joint as OJ
    from objective import entropy_weights
    pop_size = ALGO["pop_size"]
    max_iter = ALGO["max_iter"]
    n_runs = N_RUNS_PJ
    scale_keys = list(SCALES.keys())
    if args.smoke:
        max_iter, n_runs = 5, 2
        scale_keys = ["PJ1", "PJ6"]
        print(f"[冒烟] max_iter={max_iter} n_runs={n_runs} scales={scale_keys}")

    align = load_alignment()
    pc = OJ.make_plane_context(align)

    # ---- 每规模: 熵权与参考尺度(基准种群客观确定, 规模内 6 算法统一) ----
    scale_meta = {}
    for sk in scale_keys:
        step = SCALES[sk]["step_m"]
        OJ.set_profile_step(step)
        dim = OJ.DIM
        base = np.random.default_rng(2025).random((pop_size, dim))
        CE = [OJ.objectives_joint(base[i], pc)[:2] for i in range(pop_size)]
        C0 = np.array([c for c, _ in CE]); E0 = np.array([e for _, e in CE])
        wC, wE = entropy_weights(C0, E0)
        scale_meta[sk] = dict(step_m=step, dim=dim,
                              label=SCALES[sk]["label"],
                              wC=float(wC), wE=float(wE),
                              C_ref=float(C0.mean()), E_ref=float(E0.mean()))
        print(f"[规模] {SCALES[sk]['label']} dim={dim} "
              f"wC={wC:.3f} wE={wE:.3f}", flush=True)

    jobs = [(sk, algo, scale_meta[sk]["step_m"],
             scale_meta[sk]["wC"], scale_meta[sk]["wE"],
             scale_meta[sk]["C_ref"], scale_meta[sk]["E_ref"],
             pop_size, max_iter, n_runs)
            for sk in scale_keys for algo in ALGOS]
    n_workers = min(args.workers, len(jobs))
    print(f"[执行] {len(jobs)} 个(规模,算法)单元, {n_workers} 进程"
          f"(单元内 {n_runs} 次独立运行串行, 单核测时可比)", flush=True)
    with mp.Pool(n_workers) as pool:
        results = pool.map(run_unit, jobs)

    out = dict(meta=dict(pop_size=pop_size, max_iter=max_iter, n_runs=n_runs,
                         total_km=align["total_km"], algos=ALGOS,
                         scales={k: scale_meta[k] for k in scale_keys},
                         n_mode=OJ.N_MODE, corridor_half_w=OJ.CORRIDOR_HALF_W,
                         smoke=bool(args.smoke), n_workers=n_workers,
                         pen_scale=1.0,
                         execution="按(规模,算法)并行、单元内部串行(单核, 测时可比); "
                                   "平纵联合口径(问题19), 交叉桥内生(问题21)"),
               scales={sk: dict(**scale_meta[sk], algos={}, curves={})
                       for sk in scale_keys})
    for sk, algo, unit in results:
        curve = unit.pop("curve")
        out["scales"][sk]["algos"][algo] = unit
        if curve is not None:
            out["scales"][sk]["curves"][algo] = curve

    # ---- 统计检验: Wilcoxon(IJS vs 其它) + Friedman(每规模) ----
    for sk in scale_keys:
        F = {a: out["scales"][sk]["algos"][a]["best_fs"] for a in ALGOS}
        wil = {a: wilcoxon_ranksum(F["IJS"], F[a]) for a in ALGOS if a != "IJS"}
        chi2, fp, avg_rank = friedman_ranks(F)
        out["scales"][sk]["stats"] = dict(wilcoxon_vs_IJS=wil,
                                          friedman_chi2=chi2, friedman_p=fp,
                                          friedman_avg_rank=avg_rank)

    fn = "comparison_results_smoke.json" if args.smoke else "comparison_results.json"
    with open(os.path.join(RESULTS, fn), "w", encoding="utf-8") as fp:
        json.dump(out, fp, ensure_ascii=False, indent=2)
    print(f"\n[完成] 结果已保存 results/{fn}  总耗时 {(time.time()-t_all)/60:.1f} min")


if __name__ == "__main__":
    main()
