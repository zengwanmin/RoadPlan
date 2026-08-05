# -*- coding: utf-8 -*-
"""
run_comparison.py — 多算法对比主程序 (实验设计方案2 · 实验二)

在同一模型、同一熵权标量化目标 F、同约束、同参数、同运行环境下, 对比:
  IJS(本文) / JS(原型) / NSGA-II(学位论文原算法,关键对照) / GA / PSO / GWO
六组问题规模 P1(桩号步长500m) / P2(300m) / P3(100m) / P4(50m) / P5(25m) / P6(10m),
决策变量维度随步长减小而升高, 各独立运行30次。

指标(§3.4):
  标量: 最优/均值/标准差 F, 收敛代数, 运行时间
  Pareto: HV / IGD / Spacing (双目标 C-E, 由权重扫描生成前沿)
  统计: Wilcoxon 秩和 p 值, Friedman 平均秩

数据来源: 数据.xlsx (北环高速实测轨迹, 不可杜撰)
公式来源: 林坤锐学位论文 (objective.py 已逐条标注式号)
"""
import os, json, time
import numpy as np

from params import ALGO
from data_loader import load_alignment, resample_profile
from objective import (objectives, entropy_weights, make_scalar_fn,
                       make_biobj_fn)
from algorithms import run, VARIANTS
from benchmarks import run_GA, run_PSO, run_GWO, run_NSGA2
from metrics import (hypervolume_2d, igd, spacing, build_reference_front,
                     nondominated, wilcoxon_ranksum, friedman_ranks)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)

# 六组规模: 通过纵断面桩号步长控制决策变量维度 (§3.3.3 实验设置)
# 步长越小 -> 变坡点越密 -> 决策变量维度越高
SCALES = {
    "P1": dict(step_m=500.0, label="P1 (step 500 m)"),  # 桩号步长 500 m
    "P2": dict(step_m=300.0, label="P2 (step 300 m)"),  # 桩号步长 300 m
    "P3": dict(step_m=100.0, label="P3 (step 100 m)"),  # 桩号步长 100 m
    "P4": dict(step_m=50.0,  label="P4 (step 50 m)"),   # 桩号步长 50 m
    "P5": dict(step_m=25.0,  label="P5 (step 25 m)"),   # 桩号步长 25 m
    "P6": dict(step_m=10.0,  label="P6 (step 10 m)"),   # 桩号步长 10 m
}
ALGOS = ["IJS", "JS", "NSGA-II", "GA", "PSO", "GWO"]


def convergence_gen(curve, frac=0.99):
    f0, fstar = curve[0], curve[-1]
    if abs(f0 - fstar) < 1e-12:
        return 0
    target = f0 - frac * (f0 - fstar)
    below = np.where(curve <= target)[0]
    return int(below[0]) if len(below) else len(curve) - 1


def run_one_scalar(algo, f, lb, ub, pop0, max_iter, seed):
    """运行一个标量算法, 返回 (best_f, curve, runtime, best_x)。"""
    t0 = time.time()
    if algo == "IJS":
        r = run(f, lb, ub, pop0, max_iter, seed, **VARIANTS["V5_IJS"])
    elif algo == "JS":
        r = run(f, lb, ub, pop0, max_iter, seed, **VARIANTS["V1_JS"])
    elif algo == "GA":
        r = run_GA(f, lb, ub, pop0, max_iter, seed,
                   p1=0.8, p2=0.2)
    elif algo == "PSO":
        r = run_PSO(f, lb, ub, pop0, max_iter, seed,
                    w=0.8, c1=2.0, c2=2.0)
    elif algo == "GWO":
        r = run_GWO(f, lb, ub, pop0, max_iter, seed)
    else:
        raise ValueError(algo)
    return r["best_f"], r["curve"], time.time() - t0, r["best_x"]


def pareto_front_by_weights(algo, ctx, C_ref, E_ref, lb, ub, pop0,
                            max_iter, seed, n_weights=11):
    """
    权重扫描生成前沿(实验设计方案 §5.3.4 自适应权重): w1 从 0.1..0.9,
    每个权重用标量算法求一个折中解, 汇成前沿; 返回 (C_norm,E_norm) 数组。
    NSGA-II 直接用其原生前沿。
    """
    if algo == "NSGA-II":
        fbi = make_biobj_fn(ctx, C_ref, E_ref)
        rn = run_NSGA2(fbi, lb, ub, pop0, max_iter, seed)
        return nondominated(rn["front_F"])
    pts = []
    for w1 in np.linspace(0.1, 0.9, n_weights):
        f = make_scalar_fn(ctx, w1, 1 - w1, C_ref, E_ref)
        _, _, _, bx = run_one_scalar(algo, f, lb, ub, pop0, max_iter, seed)
        C, E, pen, _ = objectives(bx, ctx)
        pts.append([C / C_ref + pen / C_ref, E / E_ref + pen / E_ref])
    return nondominated(np.array(pts))


def main():
    t_all = time.time()
    pop_size = ALGO["pop_size"]
    max_iter = ALGO["max_iter"]
    n_runs = ALGO["n_runs"]

    align = load_alignment()
    out = dict(meta=dict(pop_size=pop_size, max_iter=max_iter, n_runs=n_runs,
                         total_km=align["total_km"], algos=ALGOS,
                         scales={k: v["label"] for k, v in SCALES.items()}),
               scales={})

    for sk, sc in SCALES.items():
        sta, gz = resample_profile(align, step_m=sc["step_m"])
        ctx = dict(sta=sta, gz=gz, total_len_m=align["s"][-1])
        dim = len(sta); lb, ub = np.zeros(dim), np.ones(dim)

        # 熵权法权重(基准种群客观确定, 该规模统一)
        base_rng = np.random.default_rng(2025)
        base_pop = base_rng.random((pop_size, dim))
        C0 = np.array([objectives(base_pop[i], ctx)[0] for i in range(pop_size)])
        E0 = np.array([objectives(base_pop[i], ctx)[1] for i in range(pop_size)])
        wC, wE = entropy_weights(C0, E0)
        C_ref, E_ref = float(C0.mean()), float(E0.mean())
        print(f"\n=== {sc['label']}  dim={dim}  wC={wC:.3f} wE={wE:.3f} ===")

        scale_res = dict(dim=dim, wC=wC, wE=wE, C_ref=C_ref, E_ref=E_ref,
                         algos={}, curves={}, fronts={})
        F_samples = {}   # 供统计检验: {algo:[30 best_f]}

        for algo in ALGOS:
            best_fs, conv_gens, runtimes = [], [], []
            run_curves = []
            for r in range(n_runs):
                rng = np.random.default_rng(1000 + r)
                pop0 = rng.random((pop_size, dim))
                if algo == "NSGA-II":
                    # NSGA-II 双目标: best_f 取其前沿中标量化最优点(同口径)
                    fbi = make_biobj_fn(ctx, C_ref, E_ref)
                    t0 = time.time()
                    rn = run_NSGA2(fbi, lb, ub, pop0, max_iter, 1000 + r)
                    rt = time.time() - t0
                    fr = rn["front_F"]
                    scal = wC * fr[:, 0] + wE * fr[:, 1]
                    bf = float(scal.min())
                    best_fs.append(bf)
                    conv_gens.append(max_iter)  # NSGA-II 无单点收敛曲线
                    runtimes.append(rt)
                    if r == n_runs // 2:
                        run_curves.append(None)
                else:
                    f = make_scalar_fn(ctx, wC, wE, C_ref, E_ref)
                    bf, curve, rt, bx = run_one_scalar(algo, f, lb, ub, pop0,
                                                       max_iter, 1000 + r)
                    best_fs.append(bf)
                    conv_gens.append(convergence_gen(curve))
                    runtimes.append(rt)
                    if r == n_runs // 2:
                        run_curves.append(curve.tolist())
            best_fs = np.array(best_fs)
            F_samples[algo] = best_fs.tolist()
            scale_res["algos"][algo] = dict(
                best=float(best_fs.min()), mean=float(best_fs.mean()),
                std=float(best_fs.std()), median=float(np.median(best_fs)),
                conv_gen_mean=float(np.mean(conv_gens)),
                runtime_mean=float(np.mean(runtimes)),
                best_fs=best_fs.tolist())
            if run_curves and run_curves[0] is not None:
                scale_res["curves"][algo] = run_curves[0]
            print(f"  [{algo:8s}] best={best_fs.min():.4f} mean={best_fs.mean():.4f} "
                  f"std={best_fs.std():.4f} t={np.mean(runtimes):.2f}s")

        # ---- Pareto 前沿(六组规模均详细计算, 用固定种子的权重扫描) ----
        print("  [Pareto] 权重扫描生成各算法前沿 ...")
        fronts = {}
        fseed_pop = np.random.default_rng(1500).random((pop_size, dim))
        for algo in ALGOS:
            fr = pareto_front_by_weights(algo, ctx, C_ref, E_ref, lb, ub,
                                         fseed_pop, max_iter, 1500,
                                         n_weights=11)
            fronts[algo] = fr.tolist()
        ref_front = build_reference_front([np.array(fronts[a]) for a in ALGOS])
        # HV 参考点: 所有前沿并集的最大值再放宽10%
        allpts = np.vstack([np.array(fronts[a]) for a in ALGOS])
        ref_pt = [allpts[:, 0].max() * 1.05, allpts[:, 1].max() * 1.05]
        pmet = {}
        for algo in ALGOS:
            fr = np.array(fronts[algo])
            pmet[algo] = dict(
                HV=hypervolume_2d(fr, ref_pt),
                IGD=igd(fr, ref_front),
                Spacing=spacing(fr),
                n_points=len(fr))
        scale_res["fronts"] = fronts
        scale_res["pareto_metrics"] = pmet
        scale_res["ref_point"] = ref_pt

        # ---- 统计检验(§3.4): Wilcoxon(IJS vs 其它) + Friedman ----
        wil = {a: wilcoxon_ranksum(F_samples["IJS"], F_samples[a])
               for a in ALGOS if a != "IJS"}
        chi2, fp, avg_rank = friedman_ranks(F_samples)
        scale_res["stats"] = dict(wilcoxon_vs_IJS=wil,
                                  friedman_chi2=chi2, friedman_p=fp,
                                  friedman_avg_rank=avg_rank)
        out["scales"][sk] = scale_res

    with open(os.path.join(RESULTS, "comparison_results.json"), "w",
              encoding="utf-8") as fp:
        json.dump(out, fp, ensure_ascii=False, indent=2)
    print(f"\n[完成] 结果已保存 results/comparison_results.json  总耗时 {time.time()-t_all:.1f}s")


if __name__ == "__main__":
    main()
