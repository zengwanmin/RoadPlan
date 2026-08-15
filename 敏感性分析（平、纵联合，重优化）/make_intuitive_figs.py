# -*- coding: utf-8 -*-
"""
make_intuitive_figs.py — 敏感性结果的直观展示图（独立脚本, 不改动 make_outputs.py）

输出到 figures_直观展示/:
  图D1b_E扇形与C稳健带  : 上=E 随 EV 渗透率的线族(每条线一个交通增长率 rj, 扇形张开),
                          下=同网格 126 情景的 C min-max 包络窄带 + 中位线("C 稳 E 变")
  图D11_敏感性龙卷风图  : 全部敏感因子对 E 与 C 的影响幅度水平条, 按长度排序
                          (交通量 ≫ EV ≫ 油价/电价 ≫ 走廊带 > 节能率/E_ext)

数据源: results/reopt_results.json (2026-08-12 交叉桥内生版, 237 点全可行)。
用法: python3 make_intuitive_figs.py
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results", "reopt_results.json")
FIG = os.path.join(HERE, "figures_直观展示")
os.makedirs(FIG, exist_ok=True)
YI = 1e8

for fn in ["Arial Unicode MS", "Heiti TC", "Songti SC", "STHeiti",
           "Hiragino Sans GB", "Noto Sans CJK SC", "WenQuanYi Zen Hei"]:
    try:
        font_manager.findfont(fn, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [fn]; break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150
plt.rcParams["font.size"] = 11


def _save(name):
    for ext in ("png", "pdf"):
        plt.savefig(os.path.join(FIG, f"{name}.{ext}"), bbox_inches="tight")
    plt.close()
    print(f"[图] figures_直观展示/{name}")


def load():
    with open(RES, encoding="utf-8") as f:
        return json.load(f)


# =============================================================
#  图 D1b: E 扇形线族 + C 稳健带 ("E 随情景剧变, C 纹丝不动")
# =============================================================
def fig_D1b(d):
    i1 = d["item1"]
    tr = np.array(i1["traffic_rates"], float)          # %
    ev = np.array(i1["ev_pens"], float)                # %
    C = np.array(i1["C_grid"], float) / YI             # (n_tr, n_ev) 亿元
    E = np.array(i1["E_grid"], float) / YI

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8.2, 7.4), sharex=True,
        gridspec_kw=dict(height_ratios=[3.2, 1.0], hspace=0.08))

    cmap = plt.get_cmap("YlOrRd")
    for r, rj in enumerate(tr):
        color = cmap(0.25 + 0.75 * r / max(len(tr) - 1, 1))
        ax1.plot(ev, E[r], "o-", ms=3.5, lw=1.9, color=color,
                 label=f"rj = {rj:.0f}%/yr")
    # 角点标注: E_max 与 E_min
    ax1.annotate(f"E_max = {E.max():.1f}e8 CNY\n(rj=10%, EV=0%)",
                 xy=(ev[0], E[-1, 0]), xytext=(12, -6),
                 textcoords="offset points", fontsize=9, color="#8b0000")
    ax1.annotate(f"E_min = {E.min():.1f}e8 CNY\n(rj=0, EV=100%)",
                 xy=(ev[-1], E[0, -1]), xytext=(-86, 10),
                 textcoords="offset points", fontsize=9, color="#1a6b1a")
    ax1.set_ylabel("Re-optimized life-cycle energy E (10$^8$ RMB)")
    ax1.set_title("Fig. D1b  Traffic growth x EV penetration: E fans out, C stays flat\n"
                  "(each point = one full re-optimization, w1=0.65; all 126 scenarios feasible)")
    ax1.legend(frameon=False, fontsize=9, title="Annual traffic growth",
               ncol=2, loc="upper right")
    ax1.grid(alpha=0.3)

    # 下面板: C 的 min-max 包络窄带 + 中位线(展示"±3% 窄带")
    C_min = C.min(axis=0); C_max = C.max(axis=0)
    C_med = np.median(C, axis=0)
    ax2.fill_between(ev, C_min, C_max, color="#4c72b0", alpha=0.25,
                     label=f"C envelope over all scenarios ({C.min():.1f}-{C.max():.1f}e8, +-3%)")
    ax2.plot(ev, C_med, "-", color="#4c72b0", lw=1.8, label="C median")
    ax2.set_ylim(C.min() - 2.5, C.max() + 2.5)      # 拉开纵轴, 显出"窄"
    ax2.set_xlabel("EV penetration (%)")
    ax2.set_ylabel("Cost C (10$^8$ RMB)")
    ax2.legend(frameon=False, fontsize=9, loc="upper right")
    ax2.grid(alpha=0.3)
    _save("图D1b_E扇形与C稳健带")


# =============================================================
#  图 D11: 敏感性龙卷风图 (各因子扫过其取值域时 E / C 的变化幅度)
#  统一口径: 每个因子取"其余因子在基准"的一维扫描, 条长 = max-min (亿元)。
#  item1 的两因子从二维网格取一维切片(另一因子固定基准索引 0)。
# =============================================================
def _range_from_grid(grid, axis, base_idx=0):
    """二维网格里沿 axis 扫描(另一维固定基准)的 min/max。"""
    A = np.array(grid, float) / YI
    line = A[:, base_idx] if axis == 0 else A[base_idx, :]
    return float(line.min()), float(line.max())


def fig_D11(d):
    i1, i2, i3 = d["item1"], d["item2"], d["item3"]
    factors = []   # (名称, E_min, E_max, C_min, C_max, 取值域说明)

    e0, e1 = _range_from_grid(i1["E_grid"], axis=0)      # 交通量(EV=0)
    c0, c1 = _range_from_grid(i1["C_grid"], axis=0)
    factors.append(("Traffic growth 0-10%/yr", e0, e1, c0, c1))
    e0, e1 = _range_from_grid(i1["E_grid"], axis=1)      # EV(rj=0)
    c0, c1 = _range_from_grid(i1["C_grid"], axis=1)
    factors.append(("EV penetration 0-100%", e0, e1, c0, c1))
    e0, e1 = _range_from_grid(i2["E_grid"], axis=0)      # 油价(电价=0)
    c0, c1 = _range_from_grid(i2["C_grid"], axis=0)
    factors.append(("Fuel price growth 0-5%/yr", e0, e1, c0, c1))
    e0, e1 = _range_from_grid(i2["E_grid"], axis=1)      # 电价(油价=0)
    c0, c1 = _range_from_grid(i2["C_grid"], axis=1)
    factors.append(("Elec price growth 0-5%/yr", e0, e1, c0, c1))
    e0, e1 = _range_from_grid(i3["E_grid"], axis=0)      # 节油率
    c0, c1 = _range_from_grid(i3["C_grid"], axis=0)
    factors.append(("Fuel-saving rate 0-5%", e0, e1, c0, c1))
    e0, e1 = _range_from_grid(i3["E_grid"], axis=1)      # 节能率
    c0, c1 = _range_from_grid(i3["C_grid"], axis=1)
    factors.append(("Elec-saving rate 0-5%", e0, e1, c0, c1))
    cor = d.get("corridor", [])
    if cor:
        Es = [p["E"] / YI for p in cor]; Cs = [p["C"] / YI for p in cor]
        factors.append(("Corridor half-width 200-2500 m",
                        min(Es), max(Es), min(Cs), max(Cs)))
    ext = d.get("ext_sens", [])
    if ext:
        Es = [p["E"] / YI for p in ext]; Cs = [p["C"] / YI for p in ext]
        factors.append(("Bridge extension E_ext 50-100 m",
                        min(Es), max(Es), min(Cs), max(Cs)))

    # 按 E 影响幅度排序(自上而下递减)
    factors.sort(key=lambda f: f[2] - f[1], reverse=True)
    names = [f[0] for f in factors]
    dE = [f[2] - f[1] for f in factors]
    dC = [f[4] - f[3] for f in factors]
    y = np.arange(len(factors))[::-1]

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    h = 0.36
    bE = ax.barh(y + h / 2, dE, height=h, color="#c44e52", alpha=0.88,
                 label="Impact range on energy E (max-min)")
    bC = ax.barh(y - h / 2, dC, height=h, color="#4c72b0", alpha=0.88,
                 label="Impact range on cost C (max-min)")
    for yi, v in zip(y + h / 2, dE):
        ax.text(v + 0.35, yi, f"{v:.1f}", va="center", fontsize=9,
                color="#8b1a1a")
    for yi, v in zip(y - h / 2, dC):
        ax.text(v + 0.35, yi, f"{v:.1f}", va="center", fontsize=9,
                color="#28527a")
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("Range of variation over factor domain (10$^8$ RMB; other factors at baseline)")
    ax.set_title("Fig. D11  Sensitivity tornado: traffic dominates E, cost C robust to all factors\n"
                 "(every point fully re-optimized; items 1-3 sliced at baseline of the other factor)")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    _save("图D11_敏感性龙卷风图")


def main():
    d = load()
    fig_D1b(d)
    fig_D11(d)
    print(f"[完成] 输出目录: {FIG}")


if __name__ == "__main__":
    main()
