# -*- coding: utf-8 -*-
"""
summarize_corridor_density.py — 汇总 8 组(4 走廊带 × 密度开/关)联合优化结果

同宽度下 B(密度ON) - A(密度OFF) 才是【建筑约束的真实代价】; 跨宽度看走廊带放宽
带来的收益与 Rmin 压力。诚实标注: 本线位走廊带内 Tier2 占比极低, 主导变量可能是 Rmin
而非密度约束。
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
WIDTHS = [500, 800, 1000, 1200]


def load(hw, tag):
    p = os.path.join(RES, f"joint_results_w{hw}_{tag}.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p, encoding="utf-8"))


def row(d):
    """从一组结果取 M-A(现状) 与 M-C(优化) 的关键量。"""
    A, C = d["M_A"], d["M_C"]
    return dict(
        CA=A["C"] / 1e8, EA=A["E"] / 1e8,
        C=C["C"] / 1e8, E=C["E"] / 1e8,
        L=C["L_km"], Rmin=C["Rmin"], pen=C["penalty"],
        d1=C.get("L_dense1_km", float("nan")),
        d2=C.get("L_dense2_km", float("nan")),
        dCpct=(C["C"] - A["C"]) / A["C"] * 100,
        dEpct=(C["E"] - A["E"]) / A["E"] * 100,
    )


def main():
    print("现状 M-A(所有组一致): C=%.4f亿 E=%.4f亿 L=22.462km\n" %
          (26.4061, 13.9460))
    hdr = ("走廊带", "密度", "C亿", "ΔC%", "E亿", "ΔE%", "L km",
           "Rmin", "pen", "Tier1km", "Tier2km")
    line = "| " + " | ".join(hdr) + " |"
    sep = "|" + "|".join(["---"] * len(hdr)) + "|"
    rows_md = [line, sep]
    data = {}
    for hw in WIDTHS:
        for tag, lbl in (("nodens", "OFF"), ("dens", "ON")):
            d = load(hw, tag)
            if d is None:
                rows_md.append(f"| {hw} | {lbl} | (缺) |")
                continue
            r = row(d)
            data[(hw, tag)] = r
            rows_md.append("| %d | %s | %.4f | %+.2f | %.4f | %+.2f | %.3f | "
                           "%.0f | %.2e | %.3f | %.3f |" % (
                               hw, lbl, r["C"], r["dCpct"], r["E"], r["dEpct"],
                               r["L"], r["Rmin"], r["pen"], r["d1"], r["d2"]))
    print("\n".join(rows_md))

    print("\n## 建筑约束真实代价 (同宽度 B[ON] - A[OFF])")
    print("| 走廊带 | ΔC(ON-OFF)亿 | ΔE(ON-OFF)亿 | Tier2 ON | Tier2 OFF |")
    print("|---|---|---|---|---|")
    for hw in WIDTHS:
        a = data.get((hw, "nodens"))
        b = data.get((hw, "dens"))
        if a and b:
            print("| %d | %+.4f | %+.4f | %.3f | %.3f |" % (
                hw, b["C"] - a["C"], b["E"] - a["E"], b["d2"], a["d2"]))

    with open(os.path.join(RES, "corridor_density_summary.md"), "w",
              encoding="utf-8") as f:
        f.write("# 走廊带 × 建筑密度约束 对照(8 组)\n\n")
        f.write("现状 M-A: C=26.4061亿 E=13.9460亿 L=22.462km (各组一致)\n\n")
        f.write("\n".join(rows_md) + "\n")
    print("\n[表] corridor_density_summary.md")


if __name__ == "__main__":
    main()
