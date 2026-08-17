#!/usr/bin/env python3
"""Regenerate the case data-layer figure with non-misleading W terminology."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
VERSION = HERE.parent
EXP = VERSION.parents[1]
SCHEME = EXP / "优化方案对比（平面、纵断面联合协同优化）"
sys.path.insert(0, str(SCHEME))
from data_loader import load_alignment  # type: ignore  # noqa: E402
import dem  # type: ignore  # noqa: E402

R_E = 6378137.0
a = load_alignment()
s_raw = np.asarray(a["s"])
s_plot = np.arange(0.0, float(s_raw[-1]) + 1e-9, 20.0)
X = np.interp(s_plot, s_raw, np.asarray(a["X"]))
Y = np.interp(s_plot, s_raw, np.asarray(a["Y"]))
lat0 = math.radians(float(a["lat"][0]))
lon0 = math.radians(float(a["lon"][0]))


def ll2xy(lon, lat):
    return (R_E * math.cos(lat0) * (np.radians(lon) - lon0),
            R_E * (np.radians(lat) - lat0))


b = np.load(EXP / "数据/OSM走廊带障碍物/buildings.npz", allow_pickle=False)
BX, BY = ll2xy(b["lon"], b["lat"])
tx, ty = np.gradient(X), np.gradient(Y)
tn = np.hypot(tx, ty) + 1e-12
nx, ny = -ty / tn, tx / tn
context_edges = ((X + 500 * nx, Y + 500 * ny),
                 (X - 500 * nx, Y - 500 * ny))

gx = np.linspace(-20800, 800, 720)
gy = np.linspace(-1400, 3000, 190)
DX, DY = np.meshgrid(gx, gy)
Emask = dem.ground_elev_xy(DX.ravel(), DY.ravel(),
                            float(a["lat"][0]), float(a["lon"][0]),
                            natural=True).reshape(DX.shape)

o = np.load(EXP / "数据/OSM走廊带障碍物/obstacles.npz", allow_pickle=False)
OX, OY = ll2xy(o["lines_lon"], o["lines_lat"])
offsets, kinds = o["offsets"], o["kind"]
colors = {"road": "#777777", "rail": "#7b3fa0", "water": "#2b8cbe"}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.titlesize": 12, "axes.labelsize": 11})
fig, ax = plt.subplots(figsize=(15, 5.4), dpi=180)
ax.pcolormesh(DX, DY, Emask, cmap="terrain", shading="auto", alpha=.62, zorder=0)
ax.scatter(BX, BY, s=3, color="#8b0000", alpha=.42, zorder=1)
for i, kind in enumerate(kinds):
    sl = slice(offsets[i], offsets[i + 1])
    ax.plot(OX[sl], OY[sl], color=colors[str(kind)], lw=.65, alpha=.62, zorder=2)
for edge_x, edge_y in context_edges:
    ax.plot(edge_x, edge_y, color="#1f6fd6", lw=1.8, ls="--", zorder=3)
ax.plot(X, Y, "r-", lw=2.4, zorder=4)
ax.set_aspect("equal")
ax.set_xlim(-20800, 800)
ax.set_ylim(-1400, 3000)
ax.set_xlabel("X East (m)")
ax.set_ylabel("Y North (m)")
ax.set_title("Case data fusion: DEM, buildings, transport network and reference centerline")
handles = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#8b0000", ms=6, label="building"),
    Line2D([0], [0], color="#777777", lw=2, label="road"),
    Line2D([0], [0], color="#7b3fa0", lw=2, label="rail"),
    Line2D([0], [0], color="#2b8cbe", lw=2, label="water"),
    Line2D([0], [0], color="r", lw=2, label="M-A reference centerline"),
    Line2D([0], [0], color="#1f6fd6", lw=1.8, ls="--",
           label="±500 m visualization band (not a hard bound)"),
]
ax.legend(handles=handles, loc="upper center", ncol=3, fontsize=9.5, framealpha=.92)
fig.tight_layout()
out = VERSION / "figures/case_data_layers.png"
fig.savefig(out, facecolor="white", bbox_inches="tight", pad_inches=.06)
plt.close(fig)
print(f"Wrote {out}")
