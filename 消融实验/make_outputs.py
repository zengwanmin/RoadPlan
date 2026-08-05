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

# 变体展示名与顺序
ORDER = ["V1_JS", "V2_JS+Tent", "V3_JS+Levy", "V4_JS+DE", "V5_IJS"]
# 表格用中文标签
LABEL = {"V1_JS": "V1 JS(基线)", "V2_JS+Tent": "V2 JS+Tent",
         "V3_JS+Levy": "V3 JS+Levy", "V4_JS+DE": "V4 JS+DE", "V5_IJS": "V5 IJS(完整)"}
# 图形用英文标签
LABEL_EN = {"V1_JS": "V1 JS (baseline)", "V2_JS+Tent": "V2 JS+Tent",
            "V3_JS+Levy": "V3 JS+Levy", "V4_JS+DE": "V4 JS+DE", "V5_IJS": "V5 IJS (full)"}
COLOR = {"V1_JS": "#7f7f7f", "V2_JS+Tent": "#1f77b4", "V3_JS+Levy": "#2ca02c",
         "V4_JS+DE": "#ff7f0e", "V5_IJS": "#d62728"}


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
        ("V5", "IJS(完整)", "✓", "✓", "✓"),
    ]
    hdr = ["编号", "变体名称", "Tent初始化", "Levy飞行", "差分进化DE"]
    _write_table("表A1_消融变体组件配置表", hdr, rows)


# =============================================================
#  表 A2: 各变体性能对比 (最优/均值/标准差/收敛代数/运行时间 + 相对JS提升率)
# =============================================================
def table_A2(d):
    v = d["variants"]
    base = v["V1_JS"]["mean"]
    hdr = ["变体", "最优F", "均值F", "标准差", "收敛代数", "运行时间(s)", "相对JS提升(%)"]
    rows = []
    for k in ORDER:
        r = v[k]
        impr = (base - r["mean"]) / base * 100.0
        rows.append((LABEL[k], f"{r['best']:.4f}", f"{r['mean']:.4f}",
                     f"{r['std']:.4f}", f"{r['conv_gen_mean']:.1f}",
                     f"{r['runtime_mean']:.2f}", f"{impr:+.2f}"))
    _write_table("表A2_各变体性能对比表", hdr, rows)


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
    for k in ORDER:
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
             ("V3_JS+Levy", "+Levy"), ("V4_JS+DE", "+DE"), ("V5_IJS", "IJS (full)")]
    vals = [v[k]["mean"] for k, _ in comps]
    labels = [lab for _, lab in comps]
    # 优化配色: 采用协调的蓝-青-绿渐变, 完整IJS用强调红
    bar_colors = ["#4c72b0", "#55a3c9", "#5aa469", "#e8a33d", "#c44e52"]
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
    data = [v[k]["best_fs"] for k in ORDER]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    bp = ax.boxplot(data, patch_artist=True, showmeans=True,
                    tick_labels=[LABEL_EN[k] for k in ORDER],
                    medianprops=dict(color="black"))
    for patch, k in zip(bp["boxes"], ORDER):
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
