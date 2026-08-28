# -*- coding: utf-8 -*-
"""
run_comparison.py — 多算法对比主程序 (实验设计方案2 · 实验二, 2026-08-12 升级)

【PJ1-PJ6 平纵联合规模阶梯】(待办清单2 问题19)
  在与主实验完全相同的【平纵联合】模型(objective_joint: 平面50正弦模态+纵断面
  变坡点, 准天然地面DEM, 交叉桥内生触发+生态隧道内生, ±500m 走廊带)上, 以
  【纵断面变坡点步长】生成规模阶梯(平面模态数固定50, 避免双变量混淆):

    PJ1 500m(dim≈95)  PJ2 400m(≈106)  PJ3 300m(≈125)
    PJ4 200m(≈162)    PJ5 100m(=275, 主实验口径)  PJ6 50m(≈499)

  对比算法: IJS(本文) / JS(原型) / NSGA-II(学位论文原算法) / GA / PSO / GWO,
  每算法每规模 10 次独立运行, pop=200, iter=300
  (IJS因Levy/DE阶段每代约3×NFE；图B1按统一的300代横轴展示收敛过程)。
  惩罚采用单一 pen_scale=1.0(所有算法同一目标函数, 保证公平; 不用主实验的
  两阶段软硬调度, 因其为 IJS 专用管线)。

  另在 PJ1/PJ3/PJ6 三个代表规模做 10 次独立 Pareto 质量评估: 每次运行中
  IJS/JS/GA/PSO/GWO 采用 9 点权重扫描(wC=0.1..0.9)，最大迭代300；
  NSGA-II采用原生双目标第一前沿；由全部算法、全部10次前沿的并集构造统一参考前沿与HV参考点，
  对每次独立运行分别计算 HV / IGD / Spacing。

【并行粒度: 标量按(规模,算法)，Pareto按(规模,算法,运行序号)】
  36 个 (规模,算法) 单元各占一个单核进程(BLAS 单线程), 单元内 10 次独立运行
  串行执行; 运行时间在同构单核条件下测得, 算法间耗时可比(声明测时口径)。
  每次运行的初始种群与种子由 (run 序号) 唯一确定, 与执行顺序无关。
  Pareto共3×6×10个独立单元；同一运行序号、同一权重点下各标量算法共享种子
  与初始种群，NSGA-II在每个运行序号独立运行一次。

用法:
  python3 run_comparison.py                 # 正式全量 (6规模×6算法×10次+Pareto)
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
from metrics import (hypervolume_2d, igd, spacing, build_reference_front,
                     nondominated, wilcoxon_signedrank, holm_adjust,
                     friedman_ranks)
from acceleration import evaluate_many_ordered, evaluation_workers

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
SCALAR_MAX_ITER = 300
N_RUNS_PJ = 10
PARETO_MAX_ITER = 300
PARETO_N_RUNS = 10
PARETO_SCALE_KEYS = ("PJ1", "PJ3", "PJ6")
PARETO_WEIGHTS = np.linspace(0.1, 0.9, 9)
SCALAR_SEED_BASE = 1000
PARETO_SEED_BASE = 20000
PARETO_RUN_STRIDE = 1000
NFE_PER_ITER = {"IJS": 3, "JS": 1, "NSGA-II": 1,
                "GA": 1, "PSO": 1, "GWO": 1}

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
    return OJ.make_scalar_joint(pc, wC, wE, C_ref, E_ref, pen_scale=1.0)


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


def convergence_nfe_axis(algo, pop_size, max_iter):
    """各算法收敛曲线采样点对应的累计NFE；IJS含一次Tent初始化。"""
    initial = pop_size * (2 if algo == "IJS" else 1)
    return (initial + np.arange(max_iter + 1) * pop_size * NFE_PER_ITER[algo])


def pareto_seed(run_idx, weight_idx=0):
    """同一Pareto运行、同一权重点在不同标量算法间共享的确定性种子。"""
    return PARETO_SEED_BASE + run_idx * PARETO_RUN_STRIDE + weight_idx


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
        seed = SCALAR_SEED_BASE + r
        rng = np.random.default_rng(seed)
        pop0 = rng.random((pop_size, dim))
        if algo == "NSGA-II":
            fbi = make_biobj_joint_fn(step_m, C_ref, E_ref)
            fbi(pop0[0])  # 编译/缓存预热放在统一计时范围之外
            t0 = time.time()
            rn = run_NSGA2(fbi, lb, ub, pop0, max_iter, seed,
                           scalar_weights=(wC, wE))
            rt = time.time() - t0
            fr = rn["front_F"]
            scal = wC * fr[:, 0] + wE * fr[:, 1]
            bi = int(np.argmin(scal))
            bf = float(scal[bi])
            bx = rn["front_X"][bi] if "front_X" in rn else None
            curve = rn["curve"]
        else:
            f = make_scalar_joint_fn(step_m, wC, wE, C_ref, E_ref)
            f(pop0[0])    # 各标量算法采用完全相同的预热与计时边界
            bf, curve, rt, bx = run_one_scalar(algo, f, lb, ub, pop0,
                                               max_iter, seed)
        best_fs.append(bf)
        conv_gens.append(convergence_gen(curve))
        runtimes.append(rt)
        curves.append(np.asarray(curve, dtype=float))
        # 可行性(pen==0)与进入可行域代数(问题19 加分项: 两段收敛报告)
        if bx is not None:
            C, E, pen, info = OJ.objectives_joint(np.asarray(bx), pc)
            feas.append(dict(penalty=float(pen), C=float(C), E=float(E),
                             L_km=float(info["L_km"]),
                             L_cross_km=float(info["L_cross_km"])))
        else:
            feas.append(None)
    best_fs = np.array(best_fs)
    unit = dict(
        best=float(best_fs.min()), mean=float(best_fs.mean()),
        std=float(best_fs.std(ddof=1)) if len(best_fs) > 1 else 0.0,
        median=float(np.median(best_fs)),
        conv_gen_mean=float(np.mean(conv_gens)),
        runtime_mean=float(np.mean(runtimes)),
        best_fs=best_fs.tolist(),
        runtimes=list(map(float, runtimes)),
        feas=feas,
        curves=[curve.tolist() for curve in curves],
        nfe_axis=convergence_nfe_axis(algo, pop_size, max_iter).tolist())
    print(f"  [{sk}][{algo:8s}] best={best_fs.min():.4f} "
          f"mean={best_fs.mean():.4f} "
          f"std={best_fs.std(ddof=1) if len(best_fs) > 1 else 0.0:.4f} "
          f"t={np.mean(runtimes):.1f}s", flush=True)
    return sk, algo, unit


def run_pareto_unit(job):
    """
    单个 (规模, 算法, 独立运行序号) 的 Pareto 前沿计算。

    标量算法以 9 个权重分别求折中解；各权重使用不同种子，但同一权重下
    五种标量算法共享初始种群和种子。NSGA-II直接输出原生双目标第一前沿。
    返回的目标均为含统一约束惩罚的 (C/C_ref, E/E_ref) 最小化口径。
    """
    (sk, algo, step_m, C_ref, E_ref,
     pop_size, max_iter, weights, run_idx) = job
    OJ, _ = _ctx(step_m)
    dim = OJ.DIM
    lb, ub = np.zeros(dim), np.ones(dim)
    fbi = make_biobj_joint_fn(step_m, C_ref, E_ref)

    if algo == "NSGA-II":
        seed = pareto_seed(run_idx)
        pop0 = np.random.default_rng(seed).random((pop_size, dim))
        rn = run_NSGA2(fbi, lb, ub, pop0, max_iter, seed)
        front = nondominated(rn["front_F"])
    else:
        points = []
        for idx, wC in enumerate(weights):
            seed = pareto_seed(run_idx, idx)
            pop0 = np.random.default_rng(seed).random((pop_size, dim))
            f = make_scalar_joint_fn(step_m, float(wC), float(1.0 - wC),
                                     C_ref, E_ref)
            _, _, _, bx = run_one_scalar(algo, f, lb, ub, pop0,
                                         max_iter, seed)
            points.append(fbi(np.asarray(bx)))
        front = nondominated(np.asarray(points, float))

    print(f"  [{sk}][{algo:8s}][Pareto run {run_idx + 1:02d}] "
          f"{len(front)} points", flush=True)
    return sk, algo, run_idx, front.tolist()


def pareto_quality(front_runs):
    """
    用全部算法、全部独立运行的并集建立统一参考前沿和HV参考点，随后对每个
    独立运行分别计算HV/IGD/Spacing，返回10次样本、汇总量及配对检验。
    """
    arrays = {
        algo: [nondominated(np.asarray(front, float)) for front in front_runs[algo]]
        for algo in ALGOS
    }
    all_fronts = [front for runs in arrays.values() for front in runs if len(front)]
    if not all_fronts:
        raise ValueError("Pareto质量评估没有可用前沿")
    ref_front = build_reference_front(all_fronts)
    all_points = np.vstack(all_fronts)
    maxima = all_points.max(axis=0)
    spans = np.ptp(all_points, axis=0)
    margin = np.maximum(0.05 * spans, 0.05 * np.maximum(np.abs(maxima), 1.0))
    ref_point = maxima + margin

    metrics = {}
    pooled_fronts = {}
    for algo, runs in arrays.items():
        hv_values = [float(hypervolume_2d(front, ref_point)) for front in runs]
        igd_values = [float(igd(front, ref_front)) for front in runs]
        spacing_values = [float(spacing(front)) for front in runs]
        n_points = [int(len(front)) for front in runs]
        pooled = nondominated(np.vstack([front for front in runs if len(front)]))
        pooled_fronts[algo] = pooled.tolist()

        def summary(values, name):
            values = np.asarray(values, dtype=float)
            return {
                name: values.tolist(),
                f"{name}_mean": float(np.mean(values)),
                f"{name}_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                f"{name}_median": float(np.median(values)),
                f"{name}_q25": float(np.quantile(values, 0.25)),
                f"{name}_q75": float(np.quantile(values, 0.75)),
            }

        metrics[algo] = dict(
            **summary(hv_values, "HV"),
            **summary(igd_values, "IGD"),
            **summary(spacing_values, "Spacing"),
            n_points=n_points,
            n_points_mean=float(np.mean(n_points)),
            pooled_n_points=int(len(pooled)),
        )

    metric_stats = {}
    for metric in ("HV", "IGD", "Spacing"):
        raw = {
            algo: wilcoxon_signedrank(metrics["IJS"][metric], metrics[algo][metric])
            for algo in ALGOS if algo != "IJS"
        }
        metric_stats[metric] = {
            "paired_wilcoxon_vs_IJS": raw,
            "holm_adjusted_p": holm_adjust(raw),
        }
    return (metrics, pooled_fronts, ref_front.tolist(), ref_point.tolist(),
            metric_stats)


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
    max_iter = SCALAR_MAX_ITER
    n_runs = N_RUNS_PJ
    scale_keys = list(SCALES.keys())
    pareto_max_iter = PARETO_MAX_ITER
    pareto_n_runs = PARETO_N_RUNS
    pareto_scale_keys = list(PARETO_SCALE_KEYS)
    if args.smoke:
        max_iter, n_runs = 5, 2
        scale_keys = ["PJ1", "PJ6"]
        pareto_max_iter, pareto_n_runs = 5, 2
        pareto_scale_keys = ["PJ1", "PJ6"]
        print(f"[冒烟] scalar: iter={max_iter}, runs={n_runs}, scales={scale_keys}; "
              f"Pareto: iter={pareto_max_iter}, runs={pareto_n_runs}, "
              f"scales={pareto_scale_keys}")

    align = load_alignment()
    pc = OJ.make_plane_context(align)

    # ---- 每规模: 熵权与参考尺度(基准种群客观确定, 规模内 6 算法统一) ----
    scale_meta = {}
    for sk in scale_keys:
        step = SCALES[sk]["step_m"]
        OJ.set_profile_step(step)
        dim = OJ.DIM
        base = np.random.default_rng(2025).random((pop_size, dim))
        def ce(x):
            return np.asarray(OJ.objectives_joint(x, pc)[:2], dtype=np.float64)

        CE = evaluate_many_ordered(ce, base)
        C0, E0 = CE[:, 0], CE[:, 1]
        wC, wE = entropy_weights(C0, E0)
        total_decision_variables = int(OJ.N_MODE + OJ.M_PROF)
        if total_decision_variables != int(OJ.DIM):
            raise RuntimeError(
                f"{sk}决策变量统计不一致: N_MODE+M_PROF="
                f"{total_decision_variables}, DIM={OJ.DIM}"
            )
        scale_meta[sk] = dict(step_m=step, dim=dim,
                              label=SCALES[sk]["label"],
                              plane_control_points=int(OJ.N_CTRL),
                              plane_decision_variables=int(OJ.N_MODE),
                              profile_control_points=int(OJ.M_PROF),
                              grade_segments=int(max(OJ.M_PROF - 1, 0)),
                              total_control_points=int(OJ.N_CTRL + OJ.M_PROF),
                              total_decision_variables=total_decision_variables,
                              wC=float(wC), wE=float(wE),
                              C_ref=float(C0.mean()), E_ref=float(E0.mean()))
        print(f"[规模] {SCALES[sk]['label']} dim={dim} "
              f"wC={wC:.3f} wE={wE:.3f}", flush=True)

    jobs = [(sk, algo, scale_meta[sk]["step_m"],
             scale_meta[sk]["wC"], scale_meta[sk]["wE"],
             scale_meta[sk]["C_ref"], scale_meta[sk]["E_ref"],
             pop_size, max_iter, n_runs)
            for sk in scale_keys for algo in ALGOS]
    pareto_weights = PARETO_WEIGHTS[::4] if args.smoke else PARETO_WEIGHTS
    pareto_jobs = [(sk, algo, scale_meta[sk]["step_m"],
                    scale_meta[sk]["C_ref"], scale_meta[sk]["E_ref"],
                    pop_size, pareto_max_iter, pareto_weights.tolist(), run_idx)
                   for sk in pareto_scale_keys for algo in ALGOS
                   for run_idx in range(pareto_n_runs)]
    n_workers = min(args.workers, max(len(jobs), len(pareto_jobs)))
    print(f"[执行] {len(jobs)} 个(规模,算法)单元, {n_workers} 进程"
          f"(单元内 {n_runs} 次独立运行串行, 单核测时可比)", flush=True)
    with mp.Pool(n_workers) as pool:
        results = pool.map(run_unit, jobs)
        print(f"[Pareto] {len(pareto_jobs)} 个(规模,算法,独立运行)单元, "
              f"规模={pareto_scale_keys}, 每算法每规模 {pareto_n_runs} 次, "
              f"最大迭代={pareto_max_iter}；标量算法每次 "
              f"{len(pareto_weights)} 点权重扫描", flush=True)
        pareto_results = pool.map(run_pareto_unit, pareto_jobs)

    out = dict(meta=dict(pop_size=pop_size, max_iter=max_iter, n_runs=n_runs,
                         total_km=align["total_km"], algos=ALGOS,
                         scales={k: scale_meta[k] for k in scale_keys},
                         n_mode=OJ.N_MODE, corridor_half_w=OJ.CORRIDOR_HALF_W,
                         smoke=bool(args.smoke), n_workers=n_workers,
                         evaluation_workers_per_process=evaluation_workers(),
                         pen_scale=1.0,
                         run_seeds=list(range(SCALAR_SEED_BASE,
                                              SCALAR_SEED_BASE + n_runs)),
                         nfe_per_iteration=NFE_PER_ITER,
                         pareto_max_iter=pareto_max_iter,
                         pareto_scales=pareto_scale_keys,
                         pareto_weights=pareto_weights.tolist(),
                         pareto_n_runs=pareto_n_runs,
                         pareto_seed_base=PARETO_SEED_BASE,
                         pareto_run_base_seeds=[pareto_seed(r)
                                                for r in range(pareto_n_runs)],
                         pareto_protocol=(
                             f"{pareto_scale_keys}三个代表规模，每算法每规模"
                             f"{pareto_n_runs}次独立Pareto运行，最大迭代"
                             f"{pareto_max_iter}；标量算法每次按{len(pareto_weights)}"
                             "点权重扫描，NSGA-II每次取原生"
                             "第一前沿；全部算法和全部运行共用统一参考前沿及HV参考点"
                         ),
                         execution="标量按(规模,算法)并行且单元内串行测时；Pareto按"
                                   "(规模,算法,运行序号)并行；平纵联合口径(问题19), "
                                   "交叉桥内生(问题21)"),
               scales={sk: dict(**scale_meta[sk], algos={}, curves={},
                                nfe_axes={}, pareto_front_runs=(
                                    {algo: [None] * pareto_n_runs for algo in ALGOS}
                                    if sk in pareto_scale_keys else {}
                                ))
                       for sk in scale_keys})
    for sk, algo, unit in results:
        curves = unit.pop("curves")
        nfe_axis = unit.pop("nfe_axis")
        out["scales"][sk]["algos"][algo] = unit
        out["scales"][sk]["curves"][algo] = curves
        out["scales"][sk]["nfe_axes"][algo] = nfe_axis

    for sk, algo, run_idx, front in pareto_results:
        out["scales"][sk]["pareto_front_runs"][algo][run_idx] = front
    for sk in pareto_scale_keys:
        front_runs = out["scales"][sk]["pareto_front_runs"]
        missing = [(algo, r) for algo in ALGOS for r, front in enumerate(front_runs[algo])
                   if front is None]
        if missing:
            raise RuntimeError(f"{sk}缺少Pareto独立运行结果: {missing[:5]}")
        metrics, pooled_fronts, ref_front, ref_point, pareto_stats = pareto_quality(
            front_runs)
        out["scales"][sk]["pareto_metrics"] = metrics
        out["scales"][sk]["pareto_metric_stats"] = pareto_stats
        out["scales"][sk]["fronts"] = pooled_fronts
        out["scales"][sk]["reference_front"] = ref_front
        out["scales"][sk]["ref_point"] = ref_point

    # ---- 统计检验: 配对Wilcoxon(IJS vs 其它, Holm校正) + Friedman(每规模) ----
    for sk in scale_keys:
        F = {a: out["scales"][sk]["algos"][a]["best_fs"] for a in ALGOS}
        wil = {a: wilcoxon_signedrank(F["IJS"], F[a])
               for a in ALGOS if a != "IJS"}
        chi2, fp, avg_rank = friedman_ranks(F)
        out["scales"][sk]["stats"] = dict(
                                          paired_wilcoxon_vs_IJS=wil,
                                          holm_adjusted_p_vs_IJS=holm_adjust(wil),
                                          friedman_chi2=chi2, friedman_p=fp,
                                          friedman_avg_rank=avg_rank)

    fn = "comparison_results_smoke.json" if args.smoke else "comparison_results.json"
    with open(os.path.join(RESULTS, fn), "w", encoding="utf-8") as fp:
        json.dump(out, fp, ensure_ascii=False, indent=2)
    print(f"\n[完成] 结果已保存 results/{fn}  总耗时 {(time.time()-t_all)/60:.1f} min")


if __name__ == "__main__":
    main()
