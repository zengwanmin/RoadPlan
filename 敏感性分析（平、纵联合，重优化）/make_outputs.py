# -*- coding: utf-8 -*-
"""
make_outputs.py — 由【每点重优化】结果生成全部图表 (实验四·重优化版)

图(横/纵轴、图例、图名均为英文, png+pdf 双出):
  图D1 (a) 重优化全生命周期成本 C vs EV渗透率(交通量多曲线)
       (b) 重优化车流能耗 E vs EV渗透率
  图D2 重优化能耗 E vs 油价增长率(电价多曲线)
  图D3 重优化能耗 E vs 节油率(节能多曲线)
  图D4 成本-能耗 Pareto前沿(每点重优化) + 权重三分区 + 熵权膝点
  图D5 重优化线形几何随 EV 的响应: 里程 L 与最小平曲线半径 Rmin
       (体现"每点重新优化线形", 而非固定一套方案)
  图D6 重成本(成本比例0.7-1.0)时 重优化成本C 与 能耗E 的变化曲线
  图D7 重能耗(能耗比例0.7-1.0)时 重优化成本C 与 能耗E 的变化曲线
  图D8 折中(成本/能耗比例各0.4-0.6)时 重优化成本C 与 能耗E 的变化曲线
表:
  表D1 权重前沿分区特征表(重优化)
  表D2 敏感性参数极值汇总表(重优化)
  表D3 重优化线形几何响应表(交通量0%一行, 随EV变化的 C/E/L/Rmin/wC)
"""
import os, json, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, cm

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results", "reopt_results.json")
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
YI = 1e8


def load():
    with open(RES, encoding="utf-8") as f:
        return json.load(f)


def _save(name):
    for ext in ("png", "pdf"):
        plt.savefig(os.path.join(FIG, f"{name}.{ext}"), bbox_inches="tight")
    plt.close()
    print(f"[图] {name}")


def _write_table(name, hdr, rows):
    with open(os.path.join(TAB, name + ".csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(hdr); w.writerows(rows)
    with open(os.path.join(TAB, name + ".md"), "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(hdr) + " |\n")
        f.write("|" + "|".join(["---"] * len(hdr)) + "|\n")
        for r in rows:
            f.write("| " + " | ".join(map(str, r)) + " |\n")
    print(f"[表] {name}")


# ---- 图D1: 交通量 & EV渗透率 -> (a)重优化成本 (b)重优化能耗 ----
def fig_D1(d):
    it = d["item1"]
    tr = np.array(it["traffic_rates"]); ev = np.array(it["ev_pens"])
    C = np.array(it["C_grid"]) / YI
    E = np.array(it["E_grid"]) / YI
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    colors = cm.viridis(np.linspace(0, 1, len(tr)))
    for i, t in enumerate(tr):
        axes[0].plot(ev, C[i], color=colors[i], lw=1.6, label=f"{t:.0f}%")
        axes[1].plot(ev, E[i], color=colors[i], lw=1.6, label=f"{t:.0f}%")
    axes[0].set_xlabel("EV market penetration (%)")
    axes[0].set_ylabel("Re-optimized life-cycle cost C (10^8 RMB)")
    axes[0].set_title("(a) Life-cycle cost (re-optimized at each point)")
    axes[1].set_xlabel("EV market penetration (%)")
    axes[1].set_ylabel("Re-optimized life-cycle energy E (10^8 RMB)")
    axes[1].set_title("(b) Energy consumption (re-optimized at each point)")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(title="Traffic growth rate", frameon=False, fontsize=8, ncol=2)
    fig.suptitle("Fig. D1  Re-optimized cost and energy vs traffic growth rate and EV penetration",
                 y=1.02)
    _save("图D1_交通量与EV渗透率敏感性_重优化")


# ---- 图D2: 油价 & 电价 -> 重优化能耗 ----
def fig_D2(d):
    it = d["item2"]
    fp = np.array(it["fuel_price"]); ep = np.array(it["elec_price"])
    E = np.array(it["E_grid"]) / YI
    plt.figure(figsize=(7.4, 4.9))
    colors = cm.plasma(np.linspace(0, 0.9, len(ep)))
    for j, e in enumerate(ep):
        plt.plot(fp, E[:, j], color=colors[j], lw=1.5, marker="o", ms=3, label=f"{e:.0f}%")
    plt.xlabel("Fuel price growth rate (%)")
    plt.ylabel("Re-optimized life-cycle energy E (10^8 RMB)")
    plt.title("Fig. D2  Re-optimized energy vs fuel and electricity price growth")
    plt.legend(title="Elec. price growth", frameon=False, fontsize=8, ncol=2)
    plt.grid(alpha=0.3)
    _save("图D2_油价与电价敏感性_重优化")


# ---- 图D3: 节油率 & 节能率 -> 重优化能耗 ----
def fig_D3(d):
    it = d["item3"]
    fs = np.array(it["fuel_save"]); es = np.array(it["elec_save"])
    E = np.array(it["E_grid"]) / YI
    plt.figure(figsize=(7.4, 4.9))
    colors = cm.winter(np.linspace(0, 1, len(es)))
    for j, e in enumerate(es):
        plt.plot(fs, E[:, j], color=colors[j], lw=1.5, marker="o", ms=3, label=f"{e:.0f}%")
    plt.xlabel("Fuel-saving rate (%)")
    plt.ylabel("Re-optimized life-cycle energy E (10^8 RMB)")
    plt.title("Fig. D3  Re-optimized energy vs fuel- and electricity-saving rates")
    plt.legend(title="Elec.-saving rate", frameon=False, fontsize=8, ncol=2)
    plt.grid(alpha=0.3)
    _save("图D3_节油率与电耗降低率敏感性_重优化")


# ---- 图D4: 成本-能耗 Pareto前沿(重优化) + 权重三分区 + 熵权膝点 ----
def _zone(w1):
    return "Energy-priority" if w1 < 0.3 else ("Balanced" if w1 < 0.7 else "Cost-priority")


def fig_D4(d):
    fr = d["item4"]["front"]
    w1 = np.array([p["w1"] for p in fr])
    C = np.array([p["C"] for p in fr]) / YI
    E = np.array([p["E"] for p in fr]) / YI
    zc = {"Energy-priority": "#4c72b0", "Balanced": "#55a868", "Cost-priority": "#c44e52"}
    plt.figure(figsize=(7.6, 5.3))
    order = np.argsort(E)
    plt.plot(E[order], C[order], "-", color="#999999", lw=1.0, zorder=1)
    for zone in ["Energy-priority", "Balanced", "Cost-priority"]:
        m = np.array([_zone(w) == zone for w in w1])
        if m.any():
            plt.scatter(E[m], C[m], s=70, color=zc[zone], label=zone, zorder=3,
                        edgecolor="k", linewidth=0.4)
    ep_C = d["item4"]["C_opt"] / YI; ep_E = d["item4"]["E_opt"] / YI
    plt.scatter([ep_E], [ep_C], s=220, marker="*", color="gold", edgecolor="k", zorder=5,
                label=f"Entropy-weight knee (wC={d['item4']['entropy_wC']:.3f})")
    plt.xlabel("Re-optimized life-cycle energy E (10^8 RMB)")
    plt.ylabel("Re-optimized life-cycle cost C (10^8 RMB)")
    plt.title("Fig. D4  Cost–energy Pareto front (re-optimized) with weight zones and entropy knee")
    plt.legend(frameon=False, fontsize=9); plt.grid(alpha=0.3)
    _save("图D4_成本能耗Pareto前沿与权重分区_重优化")


# ---- 图D5: 重优化线形几何随 EV 的响应 (里程 L 与最小平曲线半径 Rmin) ----
def fig_D5(d):
    it = d["item1"]
    ev = np.array(it["ev_pens"])
    tr = np.array(it["traffic_rates"])
    i0 = int(np.argmin(np.abs(tr)))            # 交通量 0% 行
    L = np.array(it["L_grid"])[i0]
    R = np.array(it["Rmin_grid"])[i0]
    fig, ax1 = plt.subplots(figsize=(8.0, 5.0))
    l1, = ax1.plot(ev, L, "o-", color="#55a868", lw=1.8, label="Re-optimized mileage L")
    ax1.set_xlabel("EV market penetration (%)")
    ax1.set_ylabel("Re-optimized mileage L (km)", color="#55a868")
    ax2 = ax1.twinx()
    l2, = ax2.plot(ev, R, "s--", color="#c44e52", lw=1.6, label="Re-optimized min. curve radius")
    ax2.set_ylabel("Min. horizontal-curve radius (m)", color="#c44e52")
    ax1.set_title("Fig. D5  Re-optimized geometry vs EV penetration (traffic growth 0%)\n"
                  "(alignment re-optimized at every point)")
    ax1.legend(handles=[l1, l2], frameon=False, fontsize=9, loc="best")
    ax1.grid(alpha=0.3)
    _save("图D5_重优化线形几何响应_EV")


# ---- 表D1: 权重前沿分区特征表(重优化) ----
def table_D1(d):
    fr = d["item4"]["front"]; E_opt = d["item4"]["E_opt"]
    hdr = ["Front point", "w1", "w2", "C (10^8 RMB)", "E (10^8 RMB)",
           "L (km)", "Rmin (m)", "ΔE vs knee (%)", "Zone"]
    rows = []
    for k, p in enumerate(fr, 1):
        dE = (p["E"] - E_opt) / E_opt * 100 if E_opt else 0.0
        rows.append([k, f"{p['w1']:.1f}", f"{p['w2']:.1f}", f"{p['C']/YI:.4f}",
                     f"{p['E']/YI:.4f}", f"{p['L_km']:.3f}", f"{p['Rmin']:.0f}",
                     f"{dE:+.2f}", _zone(p["w1"])])
    _write_table("表D1_权重前沿分区特征表_重优化", hdr, rows)


# ---- 表D2: 敏感性参数极值汇总表(重优化) ----
def table_D2(d):
    it1, it2, it3 = d["item1"], d["item2"], d["item3"]
    E2 = np.array(it2["E_grid"]); E3 = np.array(it3["E_grid"])
    fp = np.array(it2["fuel_price"]); ep = np.array(it2["elec_price"])
    i2 = np.unravel_index(np.nanargmax(E2), E2.shape)
    E3b = it3["E_base"]; drop = (E3b - E3[-1, -1]) / E3b * 100
    hdr = ["Sensitivity item", "Output", "Extremum", "At parameter"]
    rows = [
        ["① Traffic × EV penetration", "Max C (10^8 RMB)",
         f"{it1['C_max']['value']/YI:.4f}",
         f"traffic {it1['C_max']['traffic']:.0f}%, EV {it1['C_max']['ev']:.0f}%"],
        ["① Traffic × EV penetration", "Max E (10^8 RMB)",
         f"{it1['E_max']['value']/YI:.4f}",
         f"traffic {it1['E_max']['traffic']:.0f}%, EV {it1['E_max']['ev']:.0f}%"],
        ["② Fuel × elec price", "Max E (10^8 RMB)",
         f"{np.nanmax(E2)/YI:.4f}", f"fuel {fp[i2[0]]:.0f}%, elec {ep[i2[1]]:.0f}%"],
        ["③ Fuel- × elec-saving", "E drop at 5%&5% (%)",
         f"{drop:.2f}", "fuel-save 5%, elec-save 5%"],
    ]
    _write_table("表D2_敏感性参数极值汇总表_重优化", hdr, rows)


# ---- 表D3: 重优化线形几何响应表(交通量0%, 随EV) ----
def table_D3(d):
    it = d["item1"]
    ev = np.array(it["ev_pens"]); tr = np.array(it["traffic_rates"])
    i0 = int(np.argmin(np.abs(tr)))
    C = np.array(it["C_grid"])[i0]; E = np.array(it["E_grid"])[i0]
    L = np.array(it["L_grid"])[i0]; R = np.array(it["Rmin_grid"])[i0]
    W = np.array(it["wC_grid"])[i0]; P = np.array(it["pen_grid"])[i0]
    hdr = ["EV penetration", "Re-opt C (10^8)", "Re-opt E (10^8)",
           "Re-opt L (km)", "Rmin (m)", "wC (entropy)", "Penalty"]
    rows = []
    for k in range(len(ev)):
        rows.append([f"{ev[k]:.0f}%", f"{C[k]/YI:.4f}", f"{E[k]/YI:.4f}",
                     f"{L[k]:.3f}", f"{R[k]:.0f}", f"{W[k]:.3f}", f"{P[k]:.2e}"])
    _write_table("表D3_重优化线形几何响应_EV", hdr, rows)


# ---- 图D6/D7/D8: 三段权重重优化时 成本C 与 能耗E 的变化曲线 ----
def _fig_reweight(d, key, xvals_from, xlabel, title, savename):
    """通用: 画 重优化成本C(左轴) 与 能耗E(右轴) 随权重比例的变化曲线。
    xvals_from='wC' 用成本权重为横轴; ='wE' 用能耗权重为横轴。"""
    pts = d.get(key)
    if not pts:
        print(f"[图] 跳过 {savename}: 结果无 {key} (请先运行更新后的 run_reopt.py)")
        return
    pts = sorted(pts, key=lambda p: p["w1"])
    x = np.array([(p["w1"] if xvals_from == "wC" else p["wE"]) for p in pts])
    C = np.array([p["C"] for p in pts]) / YI
    E = np.array([p["E"] for p in pts]) / YI
    order = np.argsort(x); x, C, E = x[order], C[order], E[order]
    fig, ax1 = plt.subplots(figsize=(7.8, 5.0))
    l1, = ax1.plot(x, C, "o-", color="#c44e52", lw=1.9, label="Re-optimized cost C")
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel("Re-optimized life-cycle cost C (10^8 RMB)", color="#c44e52")
    ax2 = ax1.twinx()
    l2, = ax2.plot(x, E, "s--", color="#4c72b0", lw=1.7, label="Re-optimized energy E")
    ax2.set_ylabel("Re-optimized life-cycle energy E (10^8 RMB)", color="#4c72b0")
    ax1.set_title(title)
    ax1.legend(handles=[l1, l2], frameon=False, fontsize=9, loc="best")
    ax1.grid(alpha=0.3)
    _save(savename)


def fig_D6(d):
    _fig_reweight(d, "reweight_cost", "wC",
                  "Cost weight wC (cost-priority, 0.7–1.0)",
                  "Fig. D6  Re-optimized cost & energy vs cost weight (cost-priority 0.7–1.0)",
                  "图D6_重成本_成本能耗变化曲线_重优化")


def fig_D7(d):
    _fig_reweight(d, "reweight_energy", "wE",
                  "Energy weight wE (energy-priority, 0.8–1.0)",
                  "Fig. D7  Re-optimized cost & energy vs energy weight (energy-priority 0.8–1.0)",
                  "图D7_重能耗_成本能耗变化曲线_重优化")


def fig_D8(d):
    _fig_reweight(d, "reweight_balanced", "wC",
                  "Cost weight wC (balanced, 0.3–0.7)",
                  "Fig. D8  Re-optimized cost & energy vs weight (balanced, wC & wE each 0.3–0.7)",
                  "图D8_折中_成本能耗变化曲线_重优化")


def fig_D9(d):
    """图D9: 走廊带半宽敏感性 —— 重优化 C/E(左右轴) + 生态隧道长度(注记)。"""
    pts = d.get("corridor")
    if not pts:
        print("[图] 跳过 图D9: 结果无 corridor (请先运行 rerun_corridor.py)")
        return
    pts = sorted(pts, key=lambda p: p["corridor"])
    x = np.array([p["corridor"] for p in pts])
    C = np.array([p["C"] for p in pts]) / YI
    E = np.array([p["E"] for p in pts]) / YI
    T = np.array([p.get("L_eco_km", np.nan) for p in pts])
    fig, ax1 = plt.subplots(figsize=(7.8, 5.0))
    l1, = ax1.plot(x, C, "o-", color="#c44e52", lw=1.9, label="Re-optimized cost C")
    ax1.set_xlabel("Corridor half-width (m)")
    ax1.set_ylabel("Re-optimized life-cycle cost C (10^8 RMB)", color="#c44e52")
    ax1.set_xscale("log")
    ax1.set_xticks(x); ax1.set_xticklabels([f"{v:.0f}" for v in x])
    ax2 = ax1.twinx()
    l2, = ax2.plot(x, E, "s--", color="#4c72b0", lw=1.7, label="Re-optimized energy E")
    ax2.set_ylabel("Re-optimized life-cycle energy E (10^8 RMB)", color="#4c72b0")
    for xi, ci, ti in zip(x, C, T):
        ax1.annotate(f"tunnel {ti:.2f} km", (xi, ci), textcoords="offset points",
                     xytext=(0, 9), ha="center", fontsize=7.5, color="#555555")
    ax1.set_title("Fig. D9  Corridor half-width sensitivity (re-optimized, w1=0.65)")
    ax1.legend(handles=[l1, l2], frameon=False, fontsize=9, loc="best")
    ax1.grid(alpha=0.3, which="both")
    _save("图D9_走廊带半宽敏感性_重优化")


def table_D9(d):
    pts = d.get("corridor")
    if not pts:
        return
    rows = [[f"{p['corridor']:.0f}", f"{p['C']/YI:.4f}", f"{p['E']/YI:.4f}",
             f"{p['L_km']:.3f}", f"{p.get('L_eco_km', float('nan')):.2f}",
             f"{p['Rmin']:.0f}", f"{p['pen']:.2e}"]
            for p in sorted(pts, key=lambda p: p["corridor"])]
    _write_table("表D9_走廊带半宽敏感性",
                 ["Corridor half-width (m)", "C (10^8 RMB)", "E (10^8 RMB)",
                  "L (km)", "Eco tunnel (km)", "Rmin (m)", "Penalty"], rows)


def main():
    d = load()
    fig_D1(d); fig_D2(d); fig_D3(d); fig_D4(d); fig_D5(d)
    fig_D6(d); fig_D7(d); fig_D8(d); fig_D9(d)
    table_D1(d); table_D2(d); table_D3(d); table_D9(d)
    print("[完成] 全部图表已输出到 figures/ 与 tables/")


if __name__ == "__main__":
    main()
