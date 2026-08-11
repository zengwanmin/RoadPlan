# -*- coding: utf-8 -*-
"""summarize.py — 汇总 results/ 下各版本最优 C+E, 打印 MEMORY 用对比表。"""
import json, os, glob

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")

rows = []
for fp in sorted(glob.glob(os.path.join(R, "*.json"))):
    if "smoke" in fp:
        continue
    d = json.load(open(fp, encoding="utf-8"))
    b = d.get("best", d)
    base = d.get("baseline", {}).get("CE")
    if not base or "CE" not in b:
        continue
    rows.append(dict(file=os.path.basename(fp), CE=b["CE"], base=base,
                     imp=(1 - b["CE"] / base) * 100,
                     C=b.get("C"), E=b.get("E"), L=b.get("L_km"),
                     Rmin=b.get("Rmin"), pen=b.get("pen"),
                     br=b.get("L_bridge_new"), tu=b.get("L_tunnel_new")))
rows.sort(key=lambda r: r["CE"])
print("%-22s %10s %8s %9s %9s %8s %6s %6s %6s" % (
    "结果文件", "C+E(亿)", "降幅%", "C(亿)", "E(亿)", "L(km)", "Rmin", "新桥", "新隧"))
for r in rows:
    print("%-22s %10.4f %7.2f%% %9.4f %9.4f %8.3f %6.0f %6.2f %6.2f" % (
        r["file"], r["CE"] / 1e8, r["imp"], (r["C"] or 0) / 1e8,
        (r["E"] or 0) / 1e8, r["L"] or 0, r["Rmin"] or 0,
        r["br"] or 0, r["tu"] or 0))
if rows:
    best = rows[0]
    print("\n最优: %s  C+E=%.4f亿  相对基线(27.2756亿)降 %.2f%%  距-10%%目标(24.4528亿)差 %.4f亿"
          % (best["file"], best["CE"] / 1e8, best["imp"],
             (best["CE"] - 0.9 * best["base"]) / 1e8))
