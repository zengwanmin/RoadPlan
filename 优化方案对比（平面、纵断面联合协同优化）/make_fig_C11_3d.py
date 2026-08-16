# -*- coding: utf-8 -*-
"""
make_fig_C11_3d.py — 图C11: 最终方案(M-C)三维地形 mesh + 空间线形

三维面: 走廊带准天然地面 DEM(road-removal 重建)按网格采样, plot_surface 绘制;
三维线: M-C 决策点线形(平面 x,y + 设计高程 z, 来自 joint_results.json),
        路基段/交叉触发桥段/生态隧道段分色; 现状 M-A 中线作对照(灰)。
竖向做夸张显示(地形起伏 ~350m vs 平面 20km, 不夸张则看不出高程变化), 系数在图注声明。

独立脚本: python3 make_fig_C11_3d.py
输出: figures/图C11_最终方案三维mesh.png/.pdf
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["font.size"] = 10

from data_loader import load_alignment
import dem
import crossings as CR

# ---- 数据 ----
with open(os.path.join(HERE, "results", "joint_results.json"),
          encoding="utf-8") as f:
    d = json.load(f)
mc = d["M_C"]
xx = np.array(mc["plane_x"]); yy = np.array(mc["plane_y"])
zz = np.array(mc["design_z"]); sta = np.array(mc["sta"])

align = load_alignment()
lat0, lon0 = float(align["lat"][0]), float(align["lon"][0])
XA, YA = align["X"], align["Y"]
ZA = align["ground_z"]

# ---- 地形 mesh: 线形包络框外扩 800 m, 320×140 网格 ----
pad = 800.0
gx = np.linspace(min(xx.min(), XA.min()) - pad, max(xx.max(), XA.max()) + pad, 320)
gy = np.linspace(min(yy.min(), YA.min()) - pad, max(yy.max(), YA.max()) + pad, 140)
GX, GY = np.meshgrid(gx, gy)
GZ = dem.ground_elev_xy(GX.ravel(), GY.ravel(), lat0, lon0,
                        natural=True).reshape(GX.shape)

# ---- 线形分段: 生态隧道 / 交叉触发桥 / 常规路基 ----
eco = dem.eco_mask_xy(xx, yy, lat0, lon0)
cr = CR.detect_crossings(xx, yy, lat0, lon0)
keep = np.interp(cr["s"], sta, eco.astype(float)) <= 0.5 \
    if len(cr["s"]) else np.zeros(0, bool)
iv, L_cross_m = CR.bridge_intervals(cr, keep=keep)
bridge = CR.mask_from_intervals(sta, iv)

# ---- 绘图 ----
fig = plt.figure(figsize=(12.5, 7.2))
ax = fig.add_subplot(111, projection="3d")
surf = ax.plot_surface(GX / 1000.0, GY / 1000.0, GZ,
                       cmap="gist_earth", rstride=1, cstride=1,
                       linewidth=0, antialiased=True, alpha=0.92,
                       vmin=GZ.min(), vmax=GZ.max() * 1.15)

Z_LIFT = 6.0     # 线形抬升值(m), 避免与地形面 z-fighting 遮挡


def _plot_mask(mask, color, lw, label, z_off=Z_LIFT, ls="-"):
    m = np.asarray(mask, bool)
    # 找连续段逐段画, 避免跨段连线
    idx = np.where(m)[0]
    if len(idx) == 0:
        return
    breaks = np.where(np.diff(idx) > 1)[0]
    segs = np.split(idx, breaks + 1)
    for s_ in segs:
        ax.plot(xx[s_] / 1000.0, yy[s_] / 1000.0, zz[s_] + z_off,
                color=color, lw=lw, ls=ls, zorder=10)


normal = ~(eco | bridge)
_plot_mask(normal, "#d62728", 2.2, None)
_plot_mask(bridge, "#ff9500", 3.2, None)
_plot_mask(eco, "#8e44ad", 2.6, None, ls="--")
# 现状中线(灰, 贴地)
ax.plot(XA / 1000.0, YA / 1000.0, ZA + Z_LIFT, color="#555555",
        lw=1.0, alpha=0.75, zorder=9)

# ---- 视角/比例: 竖向夸张 ----
Lx = gx.max() - gx.min(); Ly = gy.max() - gy.min()
ax.set_box_aspect((Lx / 1000.0, Ly / 1000.0, 4.5))      # z 轴视觉高度
ax.view_init(elev=38, azim=-118)
ax.set_xlabel("X (km)", labelpad=10)
ax.set_ylabel("Y (km)", labelpad=10)
ax.set_zlabel("Elevation (m)")
ax.set_title("Fig. C11  Final alignment M-C on quasi-natural terrain mesh\n"
             "(vertical exaggeration for readability; bridges/tunnel colored)")
cb = fig.colorbar(surf, ax=ax, shrink=0.45, pad=0.02)
cb.set_label("Ground elevation (m)")
legend = [Line2D([0], [0], color="#d62728", lw=2.2, label="M-C at-grade"),
          Line2D([0], [0], color="#ff9500", lw=3.2,
                 label=f"Crossing-triggered bridges "
                       f"({L_cross_m/1000.0:.2f} km, {len(iv)} segments)"),
          Line2D([0], [0], color="#8e44ad", lw=2.6, ls="--",
                 label=f"Baiyun eco-tunnel ({mc['L_eco_km']:.2f} km)"),
          Line2D([0], [0], color="#555555", lw=1.0, label="M-A existing centerline")]
ax.legend(handles=legend, loc="upper left", frameon=False, fontsize=9)
for ext in ("png", "pdf"):
    plt.savefig(os.path.join(FIG, f"图C11_最终方案三维mesh.{ext}"),
                bbox_inches="tight")
print("[图] figures/图C11_最终方案三维mesh")
