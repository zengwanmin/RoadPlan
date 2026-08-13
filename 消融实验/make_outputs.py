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


# =============================================================
#  机制对齐指标(问题15, 预注册定义): 表A3 + 图A4-A6
#  数据源: results/ablation_traces.json (run(track=True), 前10种子)
# =============================================================
TRACES = os.path.join(HERE, "results", "ablation_traces.json")


def load_traces():
    if not os.path.exists(TRACES):
        print("[跳过] 无 ablation_traces.json(旧结果), 表A3/图A4-A6 不生成")
        return None
    with open(TRACES, encoding="utf-8") as f:
        return json.load(f)


def _escape_stats(tr, stall=20):
    """
    Levy 停滞逃逸率(预注册定义): best_f 连续 stall 代无改进的停滞期后,
    第一个产生改进的阶段若为 levy 则记一次逃逸。
    返回 (逃逸次数, 停滞期总数, levy改进代号列表)。
    """
    phases = ["main", "levy", "de"]
    dF = {p: np.array(tr["phase_dF"].get(p, [])) for p in phases}
    ngen = len(dF["main"])
    imp = {p: (dF[p] > 1e-15) if len(dF[p]) else np.zeros(ngen, bool)
           for p in phases}
    any_imp = np.zeros(ngen, bool)
    for p in phases:
        if len(imp[p]) == ngen:
            any_imp |= imp[p]
    esc, total = 0, 0
    run_len = 0
    for g in range(ngen):
        if any_imp[g]:
            if run_len >= stall:
                total += 1
                # 该代内的阶段顺序: main -> levy -> de, 归因给最先改进者
                if imp["main"][g]:
                    pass
                elif len(imp["levy"]) == ngen and imp["levy"][g]:
                    esc += 1
            run_len = 0
        else:
            run_len += 1
    levy_gens = list(np.where(imp["levy"])[0]) if len(dF["levy"]) else []
    return esc, total, levy_gens


def table_A3(d, td):
    """表A3: 机制对齐指标(每变体对含该机制的项计算, 值为10个插桩种子的中位数)。"""
    tr_all = td["traces"]
    pop = d["meta"]["pop_size"]
    hdr = ["变体", "Tent初始替换数/200", "Tent初始ΔF(总)",
           "Levy逃逸率(停滞≥20代)", "Levy相对ΔF占比(%)",
           "DE相对ΔF占比(%)", "DE效率ΔF/kNFE", "尾段100代DEΔF",
           "多样性@100代", "多样性@末代"]
    rows = []
    for k in _order(d):
        trs = tr_all.get(k, [])
        if not trs:
            continue
        def med(fn):
            vals = [fn(t) for t in trs]
            vals = [v for v in vals if v is not None]
            return float(np.median(vals)) if vals else float("nan")
        has_tent = "Tent" in k or k == "V5_IJS"
        has_levy = "Levy" in k or k == "V5_IJS"
        has_de = "DE" in k or k == "V5_IJS"
        n_rep = med(lambda t: t["tent_n_rep"]) if has_tent else None
        t_dF = med(lambda t: t["tent_dF"]) if has_tent else None

        def tot(t, p):
            a = t["phase_dF"].get(p, [])
            return float(np.sum(a)) if a else 0.0

        def share(t, p):
            s = sum(tot(t, q) for q in ("main", "levy", "de"))
            return 100.0 * tot(t, p) / s if s > 0 else 0.0
        esc_rate = None
        if has_levy:
            def _er(t):
                e, n, _ = _escape_stats(t)
                return e / n if n else None
            esc_rate = med(_er)
        levy_share = med(lambda t: share(t, "levy")) if has_levy else None
        de_share = med(lambda t: share(t, "de")) if has_de else None
        de_eff = med(lambda t: tot(t, "de") / (len(t["phase_dF"]["de"]) * pop / 1000.0)
                     ) if has_de else None
        de_tail = med(lambda t: float(np.sum(t["phase_dF"]["de"][-100:]))
                      ) if has_de else None
        div = trs[0]["diversity"]
        div100 = med(lambda t: t["diversity"][min(99, len(t["diversity"]) - 1)])
        divend = med(lambda t: t["diversity"][-1])
        fmt = lambda v, p=4: ("-" if v is None or (isinstance(v, float) and np.isnan(v))
                              else f"{v:.{p}f}")
        rows.append((LABEL[k], fmt(n_rep, 0), fmt(t_dF, 3),
                     fmt(esc_rate, 2), fmt(levy_share, 1),
                     fmt(de_share, 1), fmt(de_eff, 4), fmt(de_tail, 4),
                     fmt(div100, 4), fmt(divend, 4)))
    _write_table("表A3_机制对齐指标", hdr, rows)


def fig_A4(d, td):
    """图A4: 阶段改进归因——各阶段累计 ΔF 随代数(中位种子)。"""
    tr_all = td["traces"]
    keys = [k for k in ("V5_IJS", "V8_JS+Levy+DE", "V4_JS+DE", "V3_JS+Levy")
            if k in tr_all and tr_all[k]]
    fig, axes = plt.subplots(1, len(keys), figsize=(4.0 * len(keys), 4.2),
                             sharey=False)
    if len(keys) == 1:
        axes = [axes]
    for ax, k in zip(axes, keys):
        t = tr_all[k][0]
        for p, c, lab in (("main", "#7f7f7f", "JS main"),
                          ("levy", "#2ca02c", "Levy"),
                          ("de", "#ff7f0e", "DE")):
            a = t["phase_dF"].get(p, [])
            if a:
                ax.plot(np.cumsum(a), color=c, lw=1.6, label=lab)
        ax.set_title(LABEL_EN[k], fontsize=10)
        ax.set_xlabel("Iteration"); ax.grid(alpha=0.3)
    axes[0].set_ylabel("Cumulative best-F improvement")
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Fig. A4  Per-phase attribution of best-F improvement (tracked seed)")
    _save("图A4_阶段改进归因")


def fig_A5(d, td):
    """图A5: 种群多样性轨迹(质心平均距离/√dim, 插桩种子中位)。"""
    tr_all = td["traces"]
    plt.figure(figsize=(7.2, 4.8))
    for k in _order(d):
        trs = tr_all.get(k, [])
        if not trs:
            continue
        D = np.array([t["diversity"] for t in trs], dtype=float)
        med = np.median(D, axis=0)
        plt.plot(med, label=LABEL_EN[k], color=COLOR[k],
                 lw=2.0 if k == "V5_IJS" else 1.2,
                 ls="-" if k in ("V1_JS", "V5_IJS") else "--")
    plt.xlabel("Iteration"); plt.ylabel("Population diversity (mean dist to centroid / $\\sqrt{dim}$)")
    plt.title("Fig. A5  Population diversity trajectories")
    plt.legend(frameon=False, fontsize=8); plt.grid(alpha=0.3)
    _save("图A5_多样性轨迹")


def fig_A6(d, td):
    """图A6: Levy 改进时机分布(含 Levy 的变体, 全部插桩种子合并)。"""
    tr_all = td["traces"]
    keys = [k for k in _order(d)
            if ("Levy" in k or k == "V5_IJS") and tr_all.get(k)]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ngen = d["meta"]["max_iter"]
    bins = np.linspace(0, ngen, 26)
    for k in keys:
        gens = []
        for t in tr_all[k]:
            _, _, lg = _escape_stats(t)
            gens.extend(lg)
        if gens:
            ax.hist(gens, bins=bins, histtype="step", lw=1.8,
                    label=LABEL_EN[k], color=COLOR[k], density=True)
    ax.set_xlabel("Iteration of Levy-phase improvement")
    ax.set_ylabel("Density")
    ax.set_title("Fig. A6  Timing distribution of Levy-phase improvements")
    ax.legend(frameon=False, fontsize=9); ax.grid(alpha=0.3)
    _save("图A6_Levy改进时机分布")


def main():
    d = load()
    print(f"[权重] wC={d['meta']['wC']:.4f} wE={d['meta']['wE']:.4f} "
          f"dim={d['meta']['dim']} runs={d['meta']['n_runs']}")
    table_A1()
    table_A2(d)
    fig_A1(d)
    fig_A2(d)
    fig_A3(d)
    td = load_traces()
    if td is not None:
        table_A3(d, td)
        fig_A4(d, td)
        fig_A5(d, td)
        fig_A6(d, td)
    print("[完成] 全部图表已输出到 figures/ 与 tables/")


if __name__ == "__main__":
    main()
