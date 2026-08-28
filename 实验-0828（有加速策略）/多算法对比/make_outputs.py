# -*- coding: utf-8 -*-
"""
make_outputs.py — 由多算法对比结果生成实验设计方案要求的全部图表 (实验二)

表: 表B1(10次运行时间与最终F)  表B2(描述统计+配对检验+Friedman秩)
    表B3(10次HV/IGD/Spacing)  表B4(10次可行性)  表B5(PJ1–PJ6规模统计)
图: 图B1(10次逐代收敛中位数+IQR)  图B2(10次Pareto前沿及汇总前沿)
    图B3(10次运行时间均值+95%CI)  图B4(10次最终F分布)
图的横轴/纵轴/图例/图名均为英文。
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results", "comparison_results.json")
FIG = os.path.join(HERE, "figures"); TAB = os.path.join(HERE, "tables")
os.makedirs(FIG, exist_ok=True); os.makedirs(TAB, exist_ok=True)

for fn in ["Arial Unicode MS", "Heiti TC", "Songti SC", "STHeiti"]:
    try:
        font_manager.findfont(fn, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [fn]; break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150
plt.rcParams["font.size"] = 11

ALGOS = ["IJS", "JS", "NSGA-II", "GA", "PSO", "GWO"]
# PJ1-PJ6 平纵联合规模阶梯(问题19, 2026-08-12): 纵断面步长 500..50m
SCALE_KEYS = ["PJ1", "PJ2", "PJ3", "PJ4", "PJ5", "PJ6"]
SCALE_LABEL = {"PJ1": "PJ1 (500 m)", "PJ2": "PJ2 (400 m)", "PJ3": "PJ3 (300 m)",
               "PJ4": "PJ4 (200 m)", "PJ5": "PJ5 (100 m)", "PJ6": "PJ6 (50 m)"}
COLOR = {"IJS": "#c44e52", "JS": "#4c72b0", "NSGA-II": "#55a868",
         "GA": "#8172b3", "PSO": "#ccb974", "GWO": "#64b5cd"}
STYLE = {"IJS": "-", "JS": "--", "NSGA-II": "-.", "GA": ":", "PSO": "--", "GWO": "-."}
SCALAR_EXPECTED_RUNS = 10
PARETO_EXPECTED_RUNS = 10
SCALAR_EXPECTED_ITER = 300
PARETO_EXPECTED_ITER = 300
PARETO_SCALE_KEYS = ["PJ1", "PJ3", "PJ6"]


def load():
    with open(RES, encoding="utf-8") as f:
        d = json.load(f)
    n_runs = int(d.get("meta", {}).get("n_runs", 0))
    pareto_n_runs = int(d.get("meta", {}).get("pareto_n_runs", 0))
    max_iter = int(d.get("meta", {}).get("max_iter", 0))
    pareto_max_iter = int(d.get("meta", {}).get("pareto_max_iter", 0))
    pareto_scales = list(d.get("meta", {}).get("pareto_scales", []))
    if n_runs != SCALAR_EXPECTED_RUNS or pareto_n_runs != PARETO_EXPECTED_RUNS:
        raise RuntimeError(
            f"正式图表要求标量{SCALAR_EXPECTED_RUNS}次、Pareto"
            f"{PARETO_EXPECTED_RUNS}次独立运行；"
            f"当前结果为 scalar={n_runs}, Pareto={pareto_n_runs}。请先重新运行实验。"
        )
    if (max_iter != SCALAR_EXPECTED_ITER or
            pareto_max_iter != PARETO_EXPECTED_ITER):
        raise RuntimeError(
            f"正式图表要求标量与Pareto均为300代；当前结果为 "
            f"scalar={max_iter}, Pareto={pareto_max_iter}。请先重新运行实验。"
        )
    if pareto_scales != PARETO_SCALE_KEYS:
        raise RuntimeError(
            f"正式图表要求Pareto规模为{PARETO_SCALE_KEYS}；"
            f"当前结果为{pareto_scales}。请先重新运行实验。"
        )
    for s, scale in d.get("scales", {}).items():
        scale_fields = ("plane_control_points", "plane_decision_variables",
                        "profile_control_points", "grade_segments",
                        "total_control_points", "total_decision_variables", "dim")
        missing_fields = [name for name in scale_fields if name not in scale]
        if missing_fields:
            raise RuntimeError(f"{s}缺少表B5规模字段: {missing_fields}")
        if (scale["total_decision_variables"] != scale["dim"] or
                scale["total_decision_variables"] !=
                scale["plane_decision_variables"] + scale["profile_control_points"] or
                scale["total_control_points"] !=
                scale["plane_control_points"] + scale["profile_control_points"] or
                scale["profile_control_points"] - 1 != scale["grade_segments"]):
            raise RuntimeError(f"{s}表B5规模统计关系不一致")
        for algo in ALGOS:
            scalar_n = len(scale.get("algos", {}).get(algo, {}).get("best_fs", []))
            curves = scale.get("curves", {}).get(algo, [])
            nfe_axis = scale.get("nfe_axes", {}).get(algo, [])
            curve_n = len(curves)
            if (scalar_n, curve_n) != (SCALAR_EXPECTED_RUNS,) * 2:
                raise RuntimeError(
                    f"{s}/{algo}运行数不完整: final={scalar_n}, "
                    f"curves={curve_n}"
                )
            if not nfe_axis or any(len(curve) != len(nfe_axis) for curve in curves):
                raise RuntimeError(f"{s}/{algo}收敛曲线与NFE轴长度不一致")
            if (len(nfe_axis) != SCALAR_EXPECTED_ITER + 1 or
                    any(len(curve) != SCALAR_EXPECTED_ITER + 1
                        for curve in curves)):
                raise RuntimeError(
                    f"{s}/{algo}收敛曲线不是{SCALAR_EXPECTED_ITER}代"
                )
            if s in pareto_scales:
                pareto_runs = scale.get("pareto_front_runs", {}).get(algo, [])
                if len(pareto_runs) != PARETO_EXPECTED_RUNS:
                    raise RuntimeError(
                        f"{s}/{algo} Pareto运行数不是{PARETO_EXPECTED_RUNS}"
                    )
                if any(front is None for front in pareto_runs):
                    raise RuntimeError(f"{s}/{algo}存在缺失的Pareto独立运行前沿")
                metrics = scale.get("pareto_metrics", {}).get(algo, {})
                for metric in ("HV", "IGD", "Spacing"):
                    if len(metrics.get(metric, [])) != PARETO_EXPECTED_RUNS:
                        raise RuntimeError(
                            f"{s}/{algo}/{metric}不是{PARETO_EXPECTED_RUNS}次指标"
                        )
    return d


def _bootstrap_mean_ci(values, confidence=0.95, n_resamples=10000,
                       seed=20260827):
    """独立运行均值的确定性bootstrap置信区间。"""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_resamples, len(values)))
    boot = values[idx].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return tuple(np.quantile(boot, [alpha, 1.0 - alpha]))


def _write_table(name, hdr, rows):
    import csv
    with open(os.path.join(TAB, name + ".csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(hdr); w.writerows(rows)
    with open(os.path.join(TAB, name + ".md"), "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(hdr) + " |\n")
        f.write("|" + "|".join(["---"] * len(hdr)) + "|\n")
        for r in rows:
            f.write("| " + " | ".join(map(str, r)) + " |\n")
    print(f"[表] {name}")


# ---- 表B1: 六规模10次运行时间与最终综合效益F ----
def table_B1(d):
    hdr = ["Algorithm", "Runs"] \
        + [f"Runtime {s}, mean±SD (s)" for s in SCALE_KEYS] \
        + [f"Final F {s}, mean±SD" for s in SCALE_KEYS]
    rows = []
    for a in ALGOS:
        rt = []
        final_f = []
        for s in SCALE_KEYS:
            ai = d["scales"][s]["algos"][a]
            runtimes = np.asarray(ai["runtimes"], dtype=float)
            rt.append(f"{np.mean(runtimes):.2f}±{np.std(runtimes, ddof=1):.2f}")
            final_f.append(f"{ai['mean']:.4f}±{ai['std']:.4f}")
        rows.append([a, SCALAR_EXPECTED_RUNS] + rt + final_f)
    _write_table("表B1_六规模运行时间与最优效益对比表", hdr, rows)


# ---- 表B2: 10次最终F + 配对Wilcoxon/Holm + Friedman秩 ----
def table_B2(d):
    hdr = ["Algorithm", "Runs", "Best F", "Mean F", "Median F", "Std F",
           "Paired Wilcoxon p (vs IJS)", "Holm-adjusted p", "Friedman rank"]
    for s in SCALE_KEYS:
        st = d["scales"][s]["stats"]
        wil = st["paired_wilcoxon_vs_IJS"]
        holm = st["holm_adjusted_p_vs_IJS"]
        rank = st["friedman_avg_rank"]
        rows = []
        for a in ALGOS:
            ai = d["scales"][s]["algos"][a]
            p = "-" if a == "IJS" else f"{wil.get(a, float('nan')):.2e}"
            p_holm = "-" if a == "IJS" else f"{holm.get(a, float('nan')):.2e}"
            rows.append([a, SCALAR_EXPECTED_RUNS, f"{ai['best']:.4f}",
                         f"{ai['mean']:.4f}", f"{ai['median']:.4f}",
                         f"{ai['std']:.4f}", p, p_holm, f"{rank[a]:.2f}"])
        _write_table(f"表B2_{s}_算法最优均值标准差与统计检验汇总表", hdr, rows)


# ---- 表B3: PJ1/PJ3/PJ6的10次Pareto质量指标 ----
def table_B3(d):
    hdr = ["Algorithm", "Runs", "HV mean (↑)", "HV SD", "HV Holm p",
           "IGD mean (↓)", "IGD SD", "IGD Holm p",
           "Spacing mean (↓)", "Spacing SD", "Spacing Holm p",
           "Front points mean", "Pooled front points"]
    for s in PARETO_SCALE_KEYS:
        pm = d["scales"][s].get("pareto_metrics", {})
        if not pm:
            continue
        pst = d["scales"][s].get("pareto_metric_stats", {})
        rows = []
        for a in ALGOS:
            m = pm.get(a, {})
            def adjusted_p(metric):
                if a == "IJS":
                    return "-"
                value = pst.get(metric, {}).get("holm_adjusted_p", {}).get(a, np.nan)
                return f"{value:.2e}"
            rows.append([
                a, PARETO_EXPECTED_RUNS,
                f"{m.get('HV_mean', np.nan):.4f}", f"{m.get('HV_std', np.nan):.4f}",
                adjusted_p("HV"),
                f"{m.get('IGD_mean', np.nan):.4f}", f"{m.get('IGD_std', np.nan):.4f}",
                adjusted_p("IGD"),
                f"{m.get('Spacing_mean', np.nan):.4f}",
                f"{m.get('Spacing_std', np.nan):.4f}", adjusted_p("Spacing"),
                f"{m.get('n_points_mean', np.nan):.1f}",
                m.get("pooled_n_points", 0),
            ])
        _write_table(f"表B3_{s}_Pareto前沿质量指标对比表", hdr, rows)


# ---- 表B4: 可行性与两段收敛(问题19 加分项): 可行率 + 最优解工程指标 ----
def table_B4(d):
    hdr = ["Scale", "Algorithm", "Feasible runs", "Best C (1e8)", "Best E (1e8)",
           "L (km)", "L_cross bridge (km)"]
    rows = []
    for s in SCALE_KEYS:
        for a in ALGOS:
            ai = d["scales"][s]["algos"][a]
            fe = [x for x in ai.get("feas", []) if x]
            if not fe:
                continue
            nf = sum(1 for x in fe if x["penalty"] <= 1e-9)
            bi = int(np.argmin([x["C"] if x["penalty"] <= 1e-9 else np.inf
                                for x in fe]))
            b = fe[bi]
            rows.append([SCALE_LABEL[s], a, f"{nf}/{len(fe)}",
                         f"{b['C']/1e8:.2f}", f"{b['E']/1e8:.2f}",
                         f"{b['L_km']:.2f}", f"{b['L_cross_km']:.2f}"])
    if rows:
        _write_table("表B4_可行性与最优解工程指标", hdr, rows)


# ---- 表B5: PJ1-PJ6控制点、坡段与决策维数统计 ----
def table_B5(d):
    hdr = ["Scale", "Profile step (m)", "Plane control points",
           "Plane decision variables (modes)", "Profile control points",
           "Grade segments", "Total control points", "Total decision variables",
           "Total decision dimension"]
    rows = []
    for s in SCALE_KEYS:
        scale = d["scales"][s]
        rows.append([
            s, f"{scale['step_m']:.0f}",
            scale["plane_control_points"],
            scale["plane_decision_variables"],
            scale["profile_control_points"],
            scale["grade_segments"],
            scale["total_control_points"],
            scale["total_decision_variables"],
            scale["dim"],
        ])
    _write_table("表B5_PJ1-PJ6求解规模统计表", hdr, rows)


# ---- 图B1: 六规模10次逐代收敛中位数与IQR（0–300代） ----
def fig_B1(d):
    n = len(SCALE_KEYS)
    ncol = 3
    nrow = (n + ncol - 1) // ncol
    # constrained_layout 自动防止 2 行子图的标题/坐标轴相互重叠
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 4.2 * nrow),
                             constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    handles, labels = [], []
    for idx, s in enumerate(SCALE_KEYS):
        ax = axes[idx]
        curves = d["scales"][s].get("curves", {})
        x = np.arange(SCALAR_EXPECTED_ITER + 1)
        for a in ALGOS:
            C = np.asarray(curves[a], dtype=float)
            med = np.median(C, axis=0)
            q25, q75 = np.quantile(C, [0.25, 0.75], axis=0)
            ax.plot(x, med, label=a, color=COLOR[a], ls=STYLE[a],
                    lw=2.0 if a == "IJS" else 1.4)
            ax.fill_between(x, q25, q75, color=COLOR[a], alpha=0.10,
                            linewidth=0)
        ax.set_xlabel("Iteration"); ax.set_ylabel("System benefit F")
        ax.set_xlim(0, SCALAR_EXPECTED_ITER)
        ax.set_xticks(np.arange(0, SCALAR_EXPECTED_ITER + 1, 50))
        ax.set_title(f"({chr(97 + idx)}) {SCALE_LABEL.get(s, s)}"); ax.set_yscale("log")
        ax.grid(alpha=0.3)
        if not handles:                      # 收集一次图例句柄, 供全图统一显示
            handles, labels = ax.get_legend_handles_labels()
    for j in range(n, len(axes)):            # 关闭多余空子图
        axes[j].axis("off")
    # 图例统一放图底部一处(替代每个子图各画一份, 更清晰)
    fig.legend(handles, labels, loc="lower center", ncol=len(ALGOS),
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        f"Fig. B1  Pointwise median convergence with IQR over {SCALAR_EXPECTED_RUNS} "
        f"independent runs ({SCALAR_EXPECTED_ITER} iterations)"
    )
    _save("图B1_六规模迭代收敛曲线")


# ---- 图B2: 10次Pareto前沿分布 + 各算法汇总非支配前沿 ----
def fig_B2(d):
    for s in PARETO_SCALE_KEYS:
        front_runs = d["scales"][s].get("pareto_front_runs", {})
        pooled_fronts = d["scales"][s].get("fronts", {})
        if not front_runs or not pooled_fronts:
            continue
        fig, ax = plt.subplots(figsize=(7.8, 5.6))
        for a in ALGOS:
            runs = [np.asarray(fr, dtype=float) for fr in front_runs[a] if len(fr)]
            if runs:
                all_points = np.vstack(runs)
                valid = np.all(all_points > 0.0, axis=1)
                all_points = all_points[valid]
                ax.scatter(all_points[:, 1], all_points[:, 0], s=9,
                           color=COLOR[a], alpha=0.06, edgecolors="none", zorder=1)
            pooled = np.asarray(pooled_fronts[a], dtype=float)
            pooled = pooled[np.all(pooled > 0.0, axis=1)]
            if len(pooled):
                pooled = pooled[np.argsort(pooled[:, 1])]
                ax.plot(pooled[:, 1], pooled[:, 0], color=COLOR[a],
                        marker="o" if a in ("IJS", "NSGA-II") else "^",
                        ms=4.5, lw=1.8 if a == "IJS" else 1.2,
                        label=f"{a} pooled front ({len(pooled)} pts)",
                        alpha=0.95, zorder=4 if a == "IJS" else 3)
        ref = np.asarray(d["scales"][s].get("reference_front", []), dtype=float)
        ref = ref[np.all(ref > 0.0, axis=1)] if len(ref) else ref
        if len(ref):
            ref = ref[np.argsort(ref[:, 1])]
            ax.plot(ref[:, 1], ref[:, 0], color="black", ls="--", lw=1.0,
                    label="Empirical reference front", zorder=2)
        # 目标强正相关 + 基线受约束惩罚 -> 用对数坐标使各算法前沿同时可见
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("Normalized energy consumption  E  (log scale)")
        ax.set_ylabel("Normalized life-cycle cost  C  (log scale)")
        ax.set_title(
            f"Fig. B2  Pareto fronts over {PARETO_EXPECTED_RUNS} independent runs "
            f"({PARETO_EXPECTED_ITER} iterations; scale {SCALE_LABEL.get(s, s)})"
        )
        ax.legend(frameon=False, fontsize=8); ax.grid(alpha=0.3, which="both")
        fig.text(0.5, 0.01,
                 "Faint points: all run-level fronts; solid lines: pooled nondominated fronts.",
                 ha="center", fontsize=8, color="#444444")
        fig.tight_layout(rect=(0.0, 0.04, 1.0, 1.0))
        _save(f"图B2_{s}_各算法Pareto前沿分布")


# ---- 图B3: 10次运行时间均值与95% bootstrap CI ----
def fig_B3(d):
    dims = [d["scales"][s]["dim"] for s in SCALE_KEYS]
    plt.figure(figsize=(7.2, 4.8))
    for ai, a in enumerate(ALGOS):
        means, lows, highs = [], [], []
        for si, s in enumerate(SCALE_KEYS):
            values = np.asarray(d["scales"][s]["algos"][a]["runtimes"], dtype=float)
            lo, hi = _bootstrap_mean_ci(values, seed=20260827 + ai * 100 + si)
            means.append(float(np.mean(values))); lows.append(lo); highs.append(hi)
        means = np.asarray(means)
        yerr = np.vstack([means - np.asarray(lows), np.asarray(highs) - means])
        plt.errorbar(dims, means, yerr=yerr, marker="o", capsize=3,
                     color=COLOR[a], ls=STYLE[a],
                     lw=2.0 if a == "IJS" else 1.4, label=a)
    plt.xlabel("Total decision dimension (plane-mode coefficients + profile variables)")
    plt.ylabel("Mean runtime (s)")
    plt.title(
        f"Fig. B3  Runtime versus total decision dimension "
        f"(mean and 95% CI, {SCALAR_EXPECTED_RUNS} independent runs)"
    )
    plt.legend(frameon=False); plt.grid(alpha=0.3)
    _save("图B3_运行时间随规模增长趋势")


# ---- 图B4: 各算法最终F分布 (逐规模各输出一图, 10次) ----
def fig_B4(d):
    for s in SCALE_KEYS:
        data = [d["scales"][s]["algos"][a]["best_fs"] for a in ALGOS]
        fig, ax = plt.subplots(figsize=(7.6, 4.8))
        bp = ax.boxplot(data, patch_artist=True, showmeans=True,
                        tick_labels=ALGOS, medianprops=dict(color="black"))
        rng = np.random.default_rng(20260827)
        for i, (patch, a, values) in enumerate(zip(bp["boxes"], ALGOS, data), start=1):
            patch.set_facecolor(COLOR[a]); patch.set_alpha(0.65)
            jitter = rng.uniform(-0.09, 0.09, size=len(values))
            ax.scatter(i + jitter, values, s=13, color=COLOR[a], alpha=0.35,
                       edgecolors="none", zorder=3)
        ax.set_ylabel(f"System benefit F ({d['meta']['n_runs']} independent runs)")
        ax.set_xlabel("Algorithm")
        ax.set_title(
            f"Fig. B4  Final-F distributions over {SCALAR_EXPECTED_RUNS} independent runs "
            f"(scale {SCALE_LABEL.get(s, s)})"
        )
        ax.grid(axis="y", alpha=0.3)
        _save(f"图B4_{s}_各算法F值箱线图")


def _save(name):
    for ext in ("png", "pdf"):
        plt.savefig(os.path.join(FIG, f"{name}.{ext}"), bbox_inches="tight")
    plt.close()
    print(f"[图] {name}")


def main():
    d = load()
    global SCALE_KEYS, PARETO_SCALE_KEYS
    SCALE_KEYS = [k for k in SCALE_KEYS if k in d["scales"]]
    PARETO_SCALE_KEYS = [k for k in PARETO_SCALE_KEYS if k in d["scales"]]
    table_B1(d); table_B2(d); table_B3(d); table_B4(d); table_B5(d)
    fig_B1(d); fig_B2(d); fig_B3(d); fig_B4(d)
    print("[完成] 全部图表已输出到 figures/ 与 tables/")


if __name__ == "__main__":
    main()
