# -*- coding: utf-8 -*-
"""
make_fig_C8b.py — 图C8b: 能耗-成本完整 Pareto 解集(固定坐标范围版)

与图C8 同数据同样式, 但按指定范围裁剪坐标轴: E∈[12,15]、C∈[0,35](亿元),
纵轴从 0 起(展示成本全尺度, 与论文图6.7 的读图习惯一致), 横轴聚焦可行
决策区(剔除 w1≤0.15 的"高架化"退化臂)。独立脚本, 不改 make_outputs.py。
输出: figures/图C8b_能耗-成本Pareto解集_固定范围.png/.pdf
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
C_YI = 1e8
XLIM = (12.0, 15.0)     # E 亿元
YLIM = (0.0, 35.0)      # C 亿元

plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150
plt.rcParams["font.size"] = 11

with open(os.path.join(HERE, "results", "joint_results.json"),
          encoding="utf-8") as f:
    d = json.load(f)

sweep = d.get("pareto_sweep", d["pareto"])
A, MC = d["M_A"], d["M_C"]
ep = d["entropy_point"]
budget = (1.0 + ep.get("budget_tol", 0.10)) * A["C"]
C = np.array([p["C"] for p in sweep]) / C_YI
E = np.array([p["E"] for p in sweep]) / C_YI
feas = np.array([p["C"] <= budget for p in sweep])

pts = [(e, c) for e, c, f in zip(E, C, feas) if f]
front = sorted((e, c) for e, c in pts
               if not any((e2 <= e and c2 <= c) and (e2 < e or c2 < c)
                          for e2, c2 in pts))

n_out = int(np.sum((E < XLIM[0]) | (E > XLIM[1]) | (C > YLIM[1])))

plt.figure(figsize=(7.8, 5.6))
plt.scatter(E[~feas], C[~feas], s=42, marker="x", color="#999999",
            label="Excluded by budget constraint (elevated-degenerate)")
plt.scatter(E[feas], C[feas], s=46, color="#4c72b0", alpha=0.85,
            label="Feasible weight-scan solutions")
if front:
    fe, fc = zip(*front)
    plt.plot(fe, fc, "-", color="#4c72b0", lw=1.4, alpha=0.9,
             label="Non-dominated front (budget-feasible)")
plt.scatter([A["E"] / C_YI], [A["C"] / C_YI], s=130, marker="s",
            color="#333333", zorder=5, label="M-A existing")
plt.scatter([MC["E"] / C_YI], [MC["C"] / C_YI], s=210, marker="*",
            color="#c44e52", edgecolor="k", zorder=6,
            label=f"M-C decision point (w1={ep.get('w1_selected', 0):.2f})")
plt.axhline(budget / C_YI, ls="--", color="#e8a33d", lw=1.2,
            label=f"Budget constraint C = {budget/C_YI:.1f}")
plt.xlim(*XLIM); plt.ylim(*YLIM)
if n_out:
    plt.annotate(f"{n_out} degenerate solutions outside view\n"
                 f"(E<{XLIM[0]:.0f} or C>{YLIM[1]:.0f}, see Fig. C8)",
                 xy=(XLIM[0] + 0.06, YLIM[1] - 1.2), fontsize=8.5,
                 color="#777777", va="top")
plt.xlabel("Life-cycle traffic energy E (10^8 RMB)")
plt.ylabel("Life-cycle cost C (10^8 RMB)")
plt.title("Fig. C8b  Pareto solution set (fixed ranges: E 12-15, C 0-35)")
plt.legend(frameon=False, fontsize=9, loc="lower left")
plt.grid(alpha=0.3)
for ext in ("png", "pdf"):
    plt.savefig(os.path.join(FIG, f"图C8b_能耗-成本Pareto解集_固定范围.{ext}"),
                bbox_inches="tight")
print("[图] figures/图C8b_能耗-成本Pareto解集_固定范围")
