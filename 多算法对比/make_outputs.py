# -*- coding: utf-8 -*-
"""
make_outputs.py — 由多算法对比结果生成实验设计方案要求的全部图表 (实验二)

表: 表B1(六规模运行时间与最优F)  表B2(最优/均值/标准差+Wilcoxon p+Friedman秩)  表B3(HV/IGD/Spacing)
图: 图B1(六规模收敛曲线 a-f)  图B2(Pareto前沿分布)  图B3(运行时间随规模)  图B4(F值箱线图)
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
SCALE_KEYS = ["P1", "P2", "P3", "P4", "P5", "P6"]
# 各规模英文标签(用于表头与图标题, 与 run_comparison.SCALES 的桩号步长一致)
SCALE_LABEL = {"P1": "P1 (500 m)", "P2": "P2 (300 m)", "P3": "P3 (100 m)",
               "P4": "P4 (50 m)", "P5": "P5 (25 m)", "P6": "P6 (10 m)"}
COLOR = {"IJS": "#c44e52", "JS": "#4c72b0", "NSGA-II": "#55a868",
         "GA": "#8172b3", "PSO": "#ccb974", "GWO": "#64b5cd"}
STYLE = {"IJS": "-", "JS": "--", "NSGA-II": "-.", "GA": ":", "PSO": "--", "GWO": "-."}


def load():
    with open(RES, encoding="utf-8") as f:
        return json.load(f)


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


# ---- 表B1: 六规模运行时间(s) 与 最优综合效益F ----
def table_B1(d):
    hdr = ["Algorithm"] \
        + [f"Runtime {s} (s)" for s in SCALE_KEYS] \
        + [f"Best F  {s}" for s in SCALE_KEYS]
    rows = []
    for a in ALGOS:
        rt = [f"{d['scales'][s]['algos'][a]['runtime_mean']:.2f}" for s in SCALE_KEYS]
        bf = [f"{d['scales'][s]['algos'][a]['best']:.4f}" for s in SCALE_KEYS]
        rows.append([a] + rt + bf)
    _write_table("表B1_六规模运行时间与最优效益对比表", hdr, rows)


# ---- 表B2: 最优/均值/标准差 + Wilcoxon p + Friedman秩 (逐规模各输出一份) ----
def table_B2(d):
    hdr = ["Algorithm", "Best F", "Mean F", "Std F",
           "Wilcoxon p (vs IJS)", "Friedman rank"]
    for s in SCALE_KEYS:
        st = d["scales"][s]["stats"]
        wil = st["wilcoxon_vs_IJS"]; rank = st["friedman_avg_rank"]
        rows = []
        for a in ALGOS:
            ai = d["scales"][s]["algos"][a]
            p = "-" if a == "IJS" else f"{wil.get(a, float('nan')):.2e}"
            rows.append([a, f"{ai['best']:.4f}", f"{ai['mean']:.4f}",
                         f"{ai['std']:.4f}", p, f"{rank[a]:.2f}"])
        _write_table(f"表B2_{s}_算法最优均值标准差与统计检验汇总表", hdr, rows)


# ---- 表B3: Pareto质量指标 HV/IGD/Spacing (逐规模各输出一份) ----
def table_B3(d):
    hdr = ["Algorithm", "HV (↑)", "IGD (↓)", "Spacing (↓)", "Front points"]
    for s in SCALE_KEYS:
        pm = d["scales"][s].get("pareto_metrics", {})
        rows = []
        for a in ALGOS:
            m = pm.get(a, {})
            rows.append([a, f"{m.get('HV', float('nan')):.4f}",
                         f"{m.get('IGD', float('nan')):.4f}",
                         f"{m.get('Spacing', float('nan')):.4f}",
                         m.get("n_points", 0)])
        _write_table(f"表B3_{s}_Pareto前沿质量指标对比表", hdr, rows)


# ---- 图B1: 六规模收敛曲线 (a-f, 2×3子图) ----
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
        for a in ALGOS:
            if a in curves:
                c = np.array(curves[a])[:201]
                ax.plot(range(len(c)), c, label=a, color=COLOR[a],
                        ls=STYLE[a], lw=2.0 if a == "IJS" else 1.4)
        ax.set_xlabel("Iteration"); ax.set_ylabel("System benefit F")
        ax.set_title(f"({chr(97 + idx)}) {SCALE_LABEL.get(s, s)}"); ax.set_yscale("log")
        ax.grid(alpha=0.3)
        if not handles:                      # 收集一次图例句柄, 供全图统一显示
            handles, labels = ax.get_legend_handles_labels()
    for j in range(n, len(axes)):            # 关闭多余空子图
        axes[j].axis("off")
    # 图例统一放图底部一处(替代每个子图各画一份, 更清晰)
    fig.legend(handles, labels, loc="lower center", ncol=len(ALGOS),
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Fig. B1  Convergence curves of algorithms at six problem scales (first 200 generations)")
    _save("图B1_六规模迭代收敛曲线")


# ---- 图B2: Pareto前沿分布 (逐规模各输出一图), 叠加各算法 ----
def fig_B2(d):
    for s in SCALE_KEYS:
        fronts = d["scales"][s].get("fronts", {})
        if not fronts:
            continue
        plt.figure(figsize=(7.6, 5.4))
        for a in ALGOS:
            if a in fronts:
                fr = np.array(fronts[a])
                if len(fr):
                    order = np.argsort(fr[:, 1])
                    fr = fr[order]
                    mk = "o" if a in ("IJS", "NSGA-II") else "^"
                    plt.plot(fr[:, 1], fr[:, 0], marker=mk, color=COLOR[a],
                             ls="-" if a == "IJS" else "none",
                             ms=8 if a == "IJS" else 6,
                             lw=1.8 if a == "IJS" else 1.0,
                             label=f"{a} ({len(fr)} pts)", alpha=0.85,
                             zorder=5 if a == "IJS" else 3)
        # 目标强正相关 + 基线受约束惩罚 -> 用对数坐标使各算法前沿同时可见
        plt.xscale("log"); plt.yscale("log")
        plt.xlabel("Normalized energy consumption  E  (log scale)")
        plt.ylabel("Normalized life-cycle cost  C  (log scale)")
        plt.title(f"Fig. B2  Pareto fronts obtained by different algorithms (scale {SCALE_LABEL.get(s, s)})")
        plt.legend(frameon=False, title="Algorithm (front size)"); plt.grid(alpha=0.3, which="both")
        # 标注: 左下角为可行最优区
        plt.annotate("Feasible optimal region\n(low cost & low energy)",
                     xy=(0.46, 0.06), xytext=(3.0, 0.15),
                     fontsize=9, color="#c44e52",
                     arrowprops=dict(arrowstyle="->", color="#c44e52", lw=1.2))
        _save(f"图B2_{s}_各算法Pareto前沿分布")


# ---- 图B3: 运行时间随规模增长趋势 ----
def fig_B3(d):
    dims = [d["scales"][s]["dim"] for s in SCALE_KEYS]
    plt.figure(figsize=(7.2, 4.8))
    for a in ALGOS:
        rt = [d["scales"][s]["algos"][a]["runtime_mean"] for s in SCALE_KEYS]
        plt.plot(dims, rt, marker="o", color=COLOR[a], ls=STYLE[a],
                 lw=2.0 if a == "IJS" else 1.4, label=a)
    plt.xlabel("Problem dimension (number of grade-change points)")
    plt.ylabel("Mean runtime (s)")
    plt.title("Fig. B3  Runtime growth of algorithms with problem scale")
    plt.legend(frameon=False); plt.grid(alpha=0.3)
    _save("图B3_运行时间随规模增长趋势")


# ---- 图B4: 各算法F值箱线图 (逐规模各输出一图, 30次) ----
def fig_B4(d):
    for s in SCALE_KEYS:
        data = [d["scales"][s]["algos"][a]["best_fs"] for a in ALGOS]
        fig, ax = plt.subplots(figsize=(7.6, 4.8))
        bp = ax.boxplot(data, patch_artist=True, showmeans=True,
                        tick_labels=ALGOS, medianprops=dict(color="black"))
        for patch, a in zip(bp["boxes"], ALGOS):
            patch.set_facecolor(COLOR[a]); patch.set_alpha(0.65)
        ax.set_ylabel("System benefit F (30 independent runs)")
        ax.set_xlabel("Algorithm")
        ax.set_title(f"Fig. B4  Box plots of system benefit F across algorithms (scale {SCALE_LABEL.get(s, s)})")
        ax.grid(axis="y", alpha=0.3)
        _save(f"图B4_{s}_各算法F值箱线图")


def _save(name):
    for ext in ("png", "pdf"):
        plt.savefig(os.path.join(FIG, f"{name}.{ext}"), bbox_inches="tight")
    plt.close()
    print(f"[图] {name}")


def main():
    d = load()
    table_B1(d); table_B2(d); table_B3(d)
    fig_B1(d); fig_B2(d); fig_B3(d); fig_B4(d)
    print("[完成] 全部图表已输出到 figures/ 与 tables/")


if __name__ == "__main__":
    main()
