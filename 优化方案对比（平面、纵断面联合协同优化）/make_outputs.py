# -*- coding: utf-8 -*-
"""
make_outputs.py — 由【平纵联合协同优化】结果生成实验三全部图表

数据源: results/joint_results.json (run_joint.py 输出, 三方案 M-A/M-B/M-C 均为联合模型)
        results/twostage_results.json (run_twostage.py 输出, 两阶段对照, 供表C3)
表: 表C1(三模式四维指标 + M-B→M-C 变化率)  表C2(现状 M-A vs 本文 M-C 关键指标 + 变化%)
    表C3(现状/两阶段/平纵联合协同 三方案对比表)
图: 图C1(Pareto解集+熵权决策点)
    图C2(平面线形: 现状 vs 优化, 同一张图)
    图C3(纵断面线形: 现状 vs 优化, 同一张图)
    图C4(全生命周期成本分项堆积柱, 三方案)
    图C5(优化后边坡稳定性评估云图)
    图C6(平纵联合优化 IJS 收敛曲线)
    图C7(权重 wC 从 0 到 1 变化时优化方案帕累托前沿的变化趋势)
图的横轴/纵轴/图例/图名均为英文。能耗单位 元/日, 桥隧费用 0。
"""
import os, json, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results", "joint_results.json")
RES_TWO = os.path.join(HERE, "results", "twostage_results.json")  # 两阶段对照(run_twostage.py)
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

C_YI = 1e8   # 元 -> 亿元


def load():
    with open(RES, encoding="utf-8") as f:
        return json.load(f)


def _write_table(name, hdr, rows):
    with open(os.path.join(TAB, name + ".csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(hdr); w.writerows(rows)
    with open(os.path.join(TAB, name + ".md"), "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(hdr) + " |\n")
        f.write("|" + "|".join(["---"] * len(hdr)) + "|\n")
        for r in rows:
            f.write("| " + " | ".join(map(str, r)) + " |\n")
    print(f"[表] {name}")


def _pct(new, old):
    if old == 0:
        return "-"
    return f"{(new - old) / old * 100:+.2f}%"


def _save(name):
    for ext in ("png", "pdf"):
        plt.savefig(os.path.join(FIG, f"{name}.{ext}"), bbox_inches="tight")
    plt.close()
    print(f"[图] {name}")


# ---- 表C1: 三模式四维指标 + M-B→M-C 变化率 ----
def table_C1(d):
    A, B, C = d["M_A"], d["M_B"], d["M_C"]
    hdr = ["Dimension", "Metric", "M-A (existing)", "M-B (cost-only)",
           "M-C (joint bi-objective)", "M-B→M-C change"]

    def row(dim, metric, key, scale=1.0, unit=""):
        va, vb, vc = A[key] / scale, B[key] / scale, C[key] / scale
        return [dim, metric, f"{va:.4f}{unit}", f"{vb:.4f}{unit}",
                f"{vc:.4f}{unit}", _pct(C[key], B[key])]
    rows = [
        row("Economy", "Life-cycle cost C (10^8 RMB)", "C", C_YI),
        row("Economy", "  Land acquisition CR (10^8)", "CR", C_YI),
        row("Economy", "  Bridge/tunnel CB (10^8)", "CB", C_YI),
        row("Economy", "  Basic construction CS (10^8)", "CS", C_YI),
        row("Economy", "  Maintenance CQ (10^8)", "CQ", C_YI),
        row("Economy", "  Earthwork C_TU (10^8)", "C_TU", C_YI),
        row("Energy", "Life-cycle traffic energy E (10^8 RMB)", "E", C_YI),
        row("Energy", "  Fuel-vehicle E_fuel (10^8)", "E_fuel", C_YI),
        row("Energy", "  Electric-vehicle E_ele (10^8)", "E_ele", C_YI),
        row("Efficiency", "Length L (km)", "L_km"),
        row("Safety", "Slope hazard degree Q", "Q_mean"),
    ]
    _write_table("表C1_三模式四维指标对比表", hdr, rows)


# ---- 表C2: 现状 M-A vs 本文 M-C 关键指标 + 变化% ----
def table_C2(d):
    A, C = d["M_A"], d["M_C"]
    hdr = ["Metric", "M-A (existing)", "M-C (optimized)", "Change (%)"]
    rows = [
        ["Length L (km)", f"{A['L_km']:.3f}", f"{C['L_km']:.3f}", _pct(C['L_km'], A['L_km'])],
        ["Earthwork C_TU (10^8 RMB)", f"{A['C_TU']/C_YI:.4f}", f"{C['C_TU']/C_YI:.4f}", _pct(C['C_TU'], A['C_TU'])],
        ["Land acquisition CR (10^8 RMB)", f"{A['CR']/C_YI:.4f}", f"{C['CR']/C_YI:.4f}", _pct(C['CR'], A['CR'])],
        ["Life-cycle cost C (10^8 RMB)", f"{A['C']/C_YI:.4f}", f"{C['C']/C_YI:.4f}", _pct(C['C'], A['C'])],
        ["Life-cycle traffic energy E (10^8 RMB)", f"{A['E']/C_YI:.4f}", f"{C['E']/C_YI:.4f}", _pct(C['E'], A['E'])],
        ["Slope hazard Q", f"{A['Q_mean']:.3f}", f"{C['Q_mean']:.3f}", _pct(C['Q_mean'], A['Q_mean'])],
        ["Min plane radius (m)", "-", f"{C['Rmin']:.0f}", "(>=400 OK)"],
    ]
    _write_table("表C2_优化前后关键指标对比表", hdr, rows)


# ---- 表C3: 现状 / 两阶段(先平面后纵断面) / 平纵联合协同 三方案对比 ----
def table_C3(d):
    """三方案对比: 现状(M-A) vs 两阶段优化 vs 平纵联合协同优化。
    联合方案取自 joint_results.json(本文 M-C); 两阶段取自 twostage_results.json 的 M-C。
    两者桥隧长度/单位造价、成本/能耗口径、10m 步长完全一致, 仅求解方式不同(串联 vs 同时)。
    括号内为相对现状 M-A 的变化率。若无两阶段结果文件则跳过并提示。"""
    if not os.path.exists(RES_TWO):
        print("[表C3] 跳过: 未找到 twostage_results.json (请先运行 run_twostage.py 生成对照)")
        return
    with open(RES_TWO, encoding="utf-8") as f:
        dt = json.load(f)
    A = d["M_A"]              # 现状(联合与两阶段口径一致, 取联合的现状)
    TS = dt["M_C"]            # 两阶段优化方案
    JT = d["M_C"]             # 平纵联合协同优化方案(本文)

    hdr = ["Metric", "M-A (existing)", "Two-stage (plane→profile)",
           "Joint (plane+profile)", "Two-stage vs A", "Joint vs A"]

    def rowk(metric, key, scale=1.0, fmt="{:.4f}"):
        a, ts, jt = A[key] / scale, TS[key] / scale, JT[key] / scale
        return [metric, fmt.format(a), fmt.format(ts), fmt.format(jt),
                _pct(TS[key], A[key]), _pct(JT[key], A[key])]

    rows = [
        rowk("Life-cycle cost C (10^8 RMB)", "C", C_YI),
        rowk("  Land acquisition CR (10^8)", "CR", C_YI),
        rowk("  Bridge/tunnel CB (10^8)", "CB", C_YI),
        rowk("  Basic construction CS (10^8)", "CS", C_YI),
        rowk("  Maintenance CQ (10^8)", "CQ", C_YI),
        rowk("  Earthwork C_TU (10^8)", "C_TU", C_YI),
        rowk("Life-cycle traffic energy E (10^8 RMB)", "E", C_YI),
        rowk("Length L (km)", "L_km", 1.0, "{:.3f}"),
        rowk("Slope hazard Q", "Q_mean", 1.0, "{:.3f}"),
        ["Min plane radius (m)", "-", f"{TS['Rmin']:.0f}", f"{JT['Rmin']:.0f}",
         "(>=400)", "(>=400)"],
    ]
    _write_table("表C3_现状_两阶段_联合协同_三方案对比表", hdr, rows)


# ---- 图C1: Pareto 解集 + 熵权决策点 ----
def fig_C1(d):
    P = d["pareto"]
    C = np.array([p["C"] for p in P]) / C_YI
    E = np.array([p["E"] for p in P]) / C_YI
    ep = d["entropy_point"]
    plt.figure(figsize=(7.2, 5.2))
    order = np.argsort(E)
    plt.plot(E[order], C[order], "o-", color="#4c72b0", ms=6, lw=1.3,
             label="Pareto solution set (weight scan)", alpha=0.85)
    plt.scatter([ep["E"] / C_YI], [ep["C"] / C_YI], s=190, marker="*",
                color="#c44e52", zorder=5, edgecolor="k",
                label=f"Entropy-weight solution (M-C)\n(wC={ep['wC']:.3f}, wE={ep['wE']:.3f})")
    plt.xlabel("Life-cycle traffic energy E (10^8 RMB)")
    plt.ylabel("Life-cycle cost C (10^8 RMB)")
    plt.title("Fig. C1  Pareto solution set and the entropy-weight decision point")
    plt.legend(frameon=False); plt.grid(alpha=0.3)
    _save("图C1_Pareto解集与熵权决策点")


# ---- 图C2: 平面线形 现状 vs 优化 (同一张图) ----
def fig_C2(d):
    A, C = d["M_A"], d["M_C"]
    mx = np.array(d["measured"]["x"]) / 1000.0
    my = np.array(d["measured"]["y"]) / 1000.0
    ax_ = np.array(A["plane_x"]) / 1000.0; ay_ = np.array(A["plane_y"]) / 1000.0
    cx_ = np.array(C["plane_x"]) / 1000.0; cy_ = np.array(C["plane_y"]) / 1000.0
    plt.figure(figsize=(8.4, 5.4))
    plt.plot(mx, my, color="#bbbbbb", lw=0.8, ls=":", label="Measured trajectory")
    plt.plot(ax_, ay_, color="#333333", lw=1.7,
             label=f"M-A existing plane ({A['L_km']:.2f} km)")
    plt.plot(cx_, cy_, color="#c44e52", lw=2.1,
             label=f"M-C joint-optimized ({C['L_km']:.2f} km, {_pct(C['L_km'], A['L_km'])})")
    plt.scatter([cx_[0]], [cy_[0]], c="#2ca02c", s=55, zorder=6, label="Start")
    plt.scatter([cx_[-1]], [cy_[-1]], c="#8c564b", s=55, zorder=6, label="End")
    plt.xlabel("Easting X (km)"); plt.ylabel("Northing Y (km)")
    plt.title("Fig. C2  Horizontal alignment: existing (M-A) vs joint-optimized (M-C)")
    plt.legend(frameon=False); plt.grid(alpha=0.3); plt.axis("equal")
    _save("图C2_平面线形对比")


# ---- 图C3: 纵断面线形 现状 vs 优化 (同一张图) ----
def fig_C3(d):
    A, C = d["M_A"], d["M_C"]
    sa = np.array(A["sta"]) / 1000.0; gA = np.array(A["gz_new"]); zA = np.array(A["design_z"])
    sc = np.array(C["sta"]) / 1000.0; zC = np.array(C["design_z"])
    plt.figure(figsize=(9.2, 4.8))
    plt.plot(sa, gA, color="#bbbbbb", lw=1.0, ls=":", label="Ground line (measured)")
    plt.plot(sa, zA, color="#333333", lw=1.5, label="M-A existing profile")
    plt.plot(sc, zC, color="#c44e52", lw=1.9, label="M-C joint-optimized profile")
    plt.xlabel("Chainage (km)"); plt.ylabel("Elevation (m)")
    plt.title("Fig. C3  Longitudinal profile: existing (M-A) vs joint-optimized (M-C)")
    plt.legend(frameon=False); plt.grid(alpha=0.3)
    _save("图C3_纵断面线形对比")


# ---- 图C4: 全生命周期成本分项堆积柱 (M-A/M-B/M-C) ----
def fig_C4(d):
    modes = ["M_A", "M_B", "M_C"]
    labels = ["M-A (existing)", "M-B (cost-only)", "M-C (joint bi-obj.)"]
    keys = ["CR", "CB", "CS", "CQ", "C_TU"]
    names = ["Land acquisition CR", "Bridge/tunnel CB", "Basic construction CS",
             "Maintenance CQ", "Earthwork C_TU"]
    colors = ["#4c72b0", "#55a868", "#ccb974", "#8172b3", "#c44e52"]
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    x = np.arange(len(modes)); bottom = np.zeros(len(modes))
    for k, nm, col in zip(keys, names, colors):
        vals = np.array([d[m][k] / C_YI for m in modes])
        ax.bar(x, vals, bottom=bottom, label=nm, color=col, alpha=0.9,
               edgecolor="white", linewidth=0.6)
        bottom += vals
    for i, tot in enumerate(bottom):
        ax.text(i, tot, f"{tot:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Life-cycle cost (10^8 RMB)")
    ax.set_ylim(0, bottom.max() * 1.15)
    ax.set_title("Fig. C4  Composition of life-cycle cost across the three schemes")
    ax.legend(frameon=False, fontsize=9); ax.grid(axis="y", alpha=0.3)
    _save("图C4_全生命周期成本分项构成")


# ---- 图C5: 优化后边坡稳定性评估云图 (里程-危险度) ----
def fig_C5(d):
    C = d["M_C"]
    sta = np.array(C["sta"]) / 1000.0
    QC = np.array(C["Q_series"])
    fig, ax = plt.subplots(figsize=(9.5, 3.6))
    im = ax.imshow(QC[None, :], aspect="auto", cmap="RdYlGn_r",
                   extent=[sta[0], sta[-1], 0, 1], vmin=1, vmax=5)
    ax.set_yticks([])
    ax.set_xlabel("Chainage (km)")
    ax.set_title("Fig. C5  Slope-stability hazard assessment of the optimized alignment (M-C)")
    cb = fig.colorbar(im, ax=ax, orientation="vertical", pad=0.02)
    cb.set_label("Hazard degree Q  (low → high)")
    _save("图C5_边坡稳定性评估云图")


# ---- 图C7: 权重 wC 从 0 到 1 变化时优化方案帕累托前沿的变化趋势 ----
def fig_C7(d):
    P = d.get("pareto_sweep") or d.get("pareto")
    P = sorted(P, key=lambda p: p["w1"])
    w = np.array([p["w1"] for p in P])
    C = np.array([p["C"] for p in P]) / C_YI
    E = np.array([p["E"] for p in P]) / C_YI
    fig, ax = plt.subplots(figsize=(7.8, 5.4))
    # 前沿曲线(按能耗排序连线) + 按权重 wC 着色的散点
    order = np.argsort(E)
    ax.plot(E[order], C[order], "-", color="#999999", lw=1.0, zorder=1,
            label="Pareto front (weight sweep)")
    sc = ax.scatter(E, C, c=w, cmap="viridis", s=70, zorder=3,
                    edgecolor="k", linewidth=0.4, vmin=0, vmax=1)
    cb = fig.colorbar(sc, ax=ax); cb.set_label("Cost weight wC  (0 = energy-priority, 1 = cost-priority)")
    # 两端标注
    i0 = int(np.argmin(w)); i1 = int(np.argmax(w))
    ax.annotate(f"wC=0\n(energy-priority)", (E[i0], C[i0]), fontsize=8,
                textcoords="offset points", xytext=(6, 6))
    ax.annotate(f"wC=1\n(cost-priority)", (E[i1], C[i1]), fontsize=8,
                textcoords="offset points", xytext=(6, -14))
    ax.set_xlabel("Life-cycle traffic energy E (10^8 RMB)")
    ax.set_ylabel("Life-cycle cost C (10^8 RMB)")
    ax.set_title("Fig. C7  Pareto front evolution as cost weight wC varies from 0 to 1")
    ax.legend(frameon=False); ax.grid(alpha=0.3)
    _save("图C7_权重0到1帕累托前沿变化趋势")


# ---- 图C6: 平纵联合优化 IJS 收敛曲线 ----
def fig_C6(d):
    cC = np.array(d["convergence"]); cB = np.array(d.get("convergence_B", []))
    plt.figure(figsize=(7.6, 4.8))
    if cB.size:
        plt.plot(np.arange(len(cB)), cB, color="#4c72b0", lw=1.4, ls="--",
                 label="M-B cost-only (wC=1)")
    plt.plot(np.arange(len(cC)), cC, color="#c44e52", lw=1.9,
             label="M-C joint bi-objective (entropy weight)")
    plt.xlabel("Iteration"); plt.ylabel("Scalarized objective F")
    plt.yscale("log")
    plt.title("Fig. C6  Convergence of joint plane+profile optimization (IJS)")
    plt.legend(frameon=False); plt.grid(alpha=0.3, which="both")
    _save("图C6_平纵联合优化收敛曲线")


def main():
    d = load()
    table_C1(d); table_C2(d); table_C3(d)
    fig_C1(d); fig_C2(d); fig_C3(d); fig_C4(d); fig_C5(d); fig_C6(d); fig_C7(d)
    print("[完成] 全部图表已输出到 figures/ 与 tables/")


if __name__ == "__main__":
    main()
