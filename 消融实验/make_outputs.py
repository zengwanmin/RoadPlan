# -*- coding: utf-8 -*-
"""
make_outputs.py — 由消融实验结果生成实验设计方案要求的全部图表

表: 表A1(消融变体组件配置)  表A2(各变体性能对比)
图: 图A1(收敛曲线)  图A2(三组件贡献瀑布图)  图A3(30次结果箱线图)
输出: figures/*.png(+pdf矢量)  tables/*.csv + *.md
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results", "ablation_results.json")
FIG = os.path.join(HERE, "figures")
TAB = os.path.join(HERE, "tables")
os.makedirs(FIG, exist_ok=True); os.makedirs(TAB, exist_ok=True)

# ---- 中文字体 ----
for fn in ["Arial Unicode MS", "Heiti TC", "Songti SC", "STHeiti", "Hiragino Sans GB"]:
    try:
        font_manager.findfont(fn, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [fn]; break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150
plt.rcParams["font.size"] = 11

# 变体展示名与顺序(2³全因子8变体; 结果文件缺组合变体时自动跳过, 向后兼容)
ORDER = ["V1_JS", "V2_JS+Tent", "V3_JS+Levy", "V4_JS+DE",
         "V6_JS+Tent+Levy", "V7_JS+Tent+DE", "V8_JS+Levy+DE", "V5_IJS"]
LABEL = {"V1_JS": "V1 JS(基线)", "V2_JS+Tent": "V2 JS+Tent",
         "V3_JS+Levy": "V3 JS+Levy", "V4_JS+DE": "V4 JS+DE",
         "V6_JS+Tent+Levy": "V6 JS+Tent+Levy", "V7_JS+Tent+DE": "V7 JS+Tent+DE",
         "V8_JS+Levy+DE": "V8 JS+Levy+DE", "V5_IJS": "V5 IJS(完整)"}
LABEL_EN = {"V1_JS": "V1 JS (baseline)", "V2_JS+Tent": "V2 JS+Tent",
            "V3_JS+Levy": "V3 JS+Levy", "V4_JS+DE": "V4 JS+DE",
            "V6_JS+Tent+Levy": "V6 JS+Tent+Levy", "V7_JS+Tent+DE": "V7 JS+Tent+DE",
            "V8_JS+Levy+DE": "V8 JS+Levy+DE", "V5_IJS": "V5 IJS (full)"}
COLOR = {"V1_JS": "#7f7f7f", "V2_JS+Tent": "#1f77b4", "V3_JS+Levy": "#2ca02c",
         "V4_JS+DE": "#ff7f0e", "V6_JS+Tent+Levy": "#17becf",
         "V7_JS+Tent+DE": "#9467bd", "V8_JS+Levy+DE": "#8c564b",
         "V5_IJS": "#d62728"}
# 每代函数求值次数(NFE)倍数: 1 + levy(1) + de(1); tent 仅初始化一次
NFE_MULT = {"V1_JS": 1, "V2_JS+Tent": 1, "V3_JS+Levy": 2, "V4_JS+DE": 2,
            "V6_JS+Tent+Levy": 2, "V7_JS+Tent+DE": 2, "V8_JS+Levy+DE": 3,
            "V5_IJS": 3}


def _order(d):
    """结果文件中实际存在的变体(按 ORDER 排序)。"""
    return [k for k in ORDER if k in d["variants"]]


def load():
    with open(RES, encoding="utf-8") as f:
        return json.load(f)


# =============================================================
#  表 A1: 消融变体组件配置表
# =============================================================
def table_A1():
    rows = [
        ("V1", "JS(基线)", "✗", "✗", "✗"),
        ("V2", "JS + Tent", "✓", "✗", "✗"),
        ("V3", "JS + Levy", "✗", "✓", "✗"),
        ("V4", "JS + DE", "✗", "✗", "✓"),
        ("V6", "JS + Tent + Levy", "✓", "✓", "✗"),
        ("V7", "JS + Tent + DE", "✓", "✗", "✓"),
        ("V8", "JS + Levy + DE", "✗", "✓", "✓"),
        ("V5", "IJS(完整)", "✓", "✓", "✓"),
    ]
    hdr = ["编号", "变体名称", "Tent初始化", "Levy飞行", "差分进化DE"]
    _write_table("表A1_消融变体组件配置表", hdr, rows)


# =============================================================
#  表 A2: 各变体性能对比 (最优/均值/标准差/收敛代数/运行时间 + 相对JS提升率)
# =============================================================
def table_A2(d):
    from scipy import stats as _st
    v = d["variants"]
    base = v["V1_JS"]["mean"]
    v1_fs = np.array(v["V1_JS"]["best_fs"])
    v1_final = np.array(d["curves"]["V1_JS"])[-1]
    hdr = ["变体", "最优F", "均值F", "中位数F", "标准差", "F@100代", "F@300代",
           "到达JS终值代数", "自身99%收敛代数", "每代NFE(×pop)", "运行时间(s)",
           "相对JS提升(%)", "Wilcoxon p(vs JS)"]
    rows = []
    for k in _order(d):
        r = v[k]
        impr = (base - r["mean"]) / base * 100.0
        c = np.array(d["curves"][k])
        f100 = c[min(100, len(c) - 1)]
        f300 = c[min(300, len(c) - 1)]
        hit = np.where(c <= v1_final)[0]
        ttt = str(hit[0]) if len(hit) else ">500"
        if k == "V1_JS":
            pval = "-"
        else:
            _, p = _st.mannwhitneyu(v1_fs, np.array(r["best_fs"]),
                                    alternative="two-sided")
            pval = f"{p:.3g}"
        rows.append((LABEL[k], f"{r['best']:.4f}", f"{r['mean']:.4f}",
                     f"{np.median(r['best_fs']):.4f}", f"{r['std']:.4f}",
                     f"{f100:.3f}", f"{f300:.3f}", ttt,
                     f"{r['conv_gen_mean']:.1f}", NFE_MULT[k],
                     f"{r['runtime_mean']:.2f}", f"{impr:+.2f}", pval))
    _write_table("表A2_各变体性能对比表", hdr, rows)
    # 注: time-to-target 以 V1 中位曲线最终值为共同目标, 消除"相对自身99%"指标
    # 在不同起点(Tent)与不同收敛深度下的不可比问题(问题清单#1)。


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


# =============================================================
#  图 A1: 5变体收敛曲线对比 (前200代)
# =============================================================
def fig_A1(d):
    plt.figure(figsize=(7.2, 4.8))
    ncut = 200
    for k in _order(d):
        c = np.array(d["curves"][k])[:ncut + 1]
        plt.plot(range(len(c)), c, label=LABEL_EN[k], color=COLOR[k],
                 lw=2.0 if k == "V5_IJS" else 1.5,
                 ls="-" if k in ("V1_JS", "V5_IJS") else "--")
    plt.xlabel("Iteration"); plt.ylabel("System benefit F")
    plt.title("Fig. A1  Convergence curves of the five ablation variants (first 200 generations)")
    plt.legend(frameon=False); plt.grid(alpha=0.3)
    plt.yscale("log")
    _save("图A1_消融变体收敛曲线")


# =============================================================
#  图 A2: 三组件贡献瀑布图 (V1->V5 逐步添加)
# =============================================================
def fig_A2(d):
    v = d["variants"]
    # 增量分解: 以均值F为准, 展示单组件相对基线的降幅, 及完整IJS
    base = v["V1_JS"]["mean"]
    comps = [("V1_JS", "JS (baseline)"), ("V2_JS+Tent", "+Tent"),
             ("V3_JS+Levy", "+Levy"), ("V4_JS+DE", "+DE"),
             ("V6_JS+Tent+Levy", "+Tent+Levy"), ("V7_JS+Tent+DE", "+Tent+DE"),
             ("V8_JS+Levy+DE", "+Levy+DE"), ("V5_IJS", "IJS (full)")]
    comps = [(k, lab) for k, lab in comps if k in v]     # 向后兼容旧结果
    vals = [v[k]["mean"] for k, _ in comps]
    labels = [lab for _, lab in comps]
    bar_colors = ["#4c72b0", "#55a3c9", "#5aa469", "#e8a33d",
                  "#17becf", "#9467bd", "#8c564b", "#c44e52"][:len(comps)]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    x = np.arange(len(comps))
    bars = ax.bar(x, vals, color=bar_colors, alpha=0.9,
                  edgecolor="white", linewidth=0.8, zorder=3)
    for b, val in zip(bars, vals):
        red = (base - val) / base * 100
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                f"{val:.3f}\n({red:+.1f}%)", ha="center", va="bottom",
                fontsize=9, zorder=4)
    ax.axhline(base, ls=":", color="gray", lw=1, label="JS baseline level", zorder=2)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Mean system benefit F")
    ax.set_title("Fig. A2  Contribution decomposition of the three improvement components")
    # 上框上扩, 使数据标注完整落入图框内
    ax.set_ylim(0, max(vals) * 1.22)
    ax.legend(frameon=False, loc="upper right"); ax.grid(axis="y", alpha=0.3, zorder=0)
    _save("图A2_三组件贡献分解")


# =============================================================
#  图 A3: 30次独立运行结果箱线图
# =============================================================
def fig_A3(d):
    v = d["variants"]
    keys = _order(d)
    data = [v[k]["best_fs"] for k in keys]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    bp = ax.boxplot(data, patch_artist=True, showmeans=True,
                    tick_labels=[LABEL_EN[k] for k in keys],
                    medianprops=dict(color="black"))
    for patch, k in zip(bp["boxes"], keys):
        patch.set_facecolor(COLOR[k]); patch.set_alpha(0.6)
    ax.set_ylabel("System benefit F (30 independent runs)")
    ax.set_title("Fig. A3  Box plots of 30 independent runs for each ablation variant")
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=15)
    _save("图A3_消融变体箱线图")


def _save(name):
    for ext in ("png", "pdf"):
        plt.savefig(os.path.join(FIG, f"{name}.{ext}"), bbox_inches="tight")
    plt.close()
    print(f"[图] {name}")


def main():
    d = load()
    print(f"[权重] wC={d['meta']['wC']:.4f} wE={d['meta']['wE']:.4f} "
          f"dim={d['meta']['dim']} runs={d['meta']['n_runs']}")
    table_A1()
    table_A2(d)
    fig_A1(d)
    fig_A2(d)
    fig_A3(d)
    print("[完成] 全部图表已输出到 figures/ 与 tables/")


if __name__ == "__main__":
    main()
