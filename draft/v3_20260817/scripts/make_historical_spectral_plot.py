#!/usr/bin/env python3
"""Redraw the legacy W sensitivity without calling W a hard corridor width."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
VERSION = HERE.parent
EXP = VERSION.parents[1]
SRC = EXP / "敏感性分析（平、纵联合，重优化）/tables/表D9_走廊带半宽敏感性.csv"

with SRC.open(encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))
w = [float(r["Corridor half-width (m)"]) for r in rows]
c = [float(r["C (10^8 RMB)"]) for r in rows]
e = [float(r["E (10^8 RMB)"]) for r in rows]
tunnel = [float(r["Eco tunnel (km)"]) for r in rows]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.titlesize": 11, "axes.labelsize": 10})
fig, ax = plt.subplots(figsize=(7.2, 4.5), dpi=180)
ax2 = ax.twinx()
p1 = ax.plot(w, c, "o-", lw=2, color="#c84e4e", label="Historical re-optimized C")
p2 = ax2.plot(w, e, "s--", lw=2, color="#4775b8", label="Historical re-optimized E")
ax.set_xlabel(r"Spectral width parameter $W$ (m)")
ax.set_ylabel(r"Life-cycle cost $C$ ($10^8$ RMB)", color="#c84e4e")
ax2.set_ylabel(r"Monetized energy $E$ ($10^8$ RMB)", color="#4775b8")
ax.tick_params(axis="y", colors="#c84e4e")
ax2.tick_params(axis="y", colors="#4775b8")
ax.grid(alpha=.25)
ax.set_title("Historical free-endpoint spectral-width sensitivity")
for x, y, lt in zip(w, c, tunnel):
    ax.annotate(f"tunnel {lt:.2f} km", (x, y), xytext=(0, -14),
                textcoords="offset points", ha="center", fontsize=7.5, color="#555")
ax.legend(p1 + p2, [q.get_label() for q in p1 + p2], frameon=False, loc="upper right")
fig.text(.5, .01, "Exploratory legacy model; W bounds spectral coefficients and is not a pointwise offset limit.",
         ha="center", fontsize=8.5, color="#444")
fig.tight_layout(rect=(0, .04, 1, 1))
for suffix in ("pdf", "png"):
    fig.savefig(VERSION / f"figures/corridor_sensitivity.{suffix}",
                bbox_inches="tight", pad_inches=.06)
plt.close(fig)
print(f"Wrote corrected spectral-width plot under {VERSION / 'figures'}")
