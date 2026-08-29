# -*- coding: utf-8 -*-
"""
make_fig_C11_3d.py — 图C11: 最终方案(M-C)三维地形 mesh + 空间线形(出版级样式)

三维面: 走廊带准天然地面 DEM(road-removal 重建), 480×200 网格 + LightSource
        山体阴影晕渲(hillshade, soft blend), 浅色地形配色使线形前景突出;
三维线: M-C 线形(平面 x,y + 设计高程 z), 路基/交叉触发桥/生态隧道分色,
        白色描边(halo)保证任何底色上可读; M-A 现状中线作对照。
坐标: 以包络框西南角为原点的公里坐标; 竖向夸张系数在图注明示。

独立脚本: python3 make_fig_C11_3d.py
输出: figures/图C11_最终方案三维mesh.png/.pdf
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LightSource, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
plt.rcParams.update({
    "figure.dpi": 200, "font.size": 9.5, "font.family": "DejaVu Sans",
    "axes.linewidth": 0.6, "mathtext.default": "regular",
})

from data_loader import load_alignment
import dem
import crossings as CR

# ---------------- 数据 ----------------
with open(os.path.join(HERE, "results", "joint_results.json"),
          encoding="utf-8") as f:
    d = json.load(f)
mc = d["M_C"]
xx = np.array(mc["plane_x"]); yy = np.array(mc["plane_y"])
zz = np.array(mc["design_z"]); sta = np.array(mc["sta"])

align = load_alignment()
lat0, lon0 = float(align["lat"][0]), float(align["lon"][0])
XA, YA, ZA = align["X"], align["Y"], align["ground_z"]

pad = 700.0
x0 = min(xx.min(), XA.min()) - pad; x1 = max(xx.max(), XA.max()) + pad
y0 = min(yy.min(), YA.min()) - pad; y1 = max(yy.max(), YA.max()) + pad
gx = np.linspace(x0, x1, 480); gy = np.linspace(y0, y1, 200)
GX, GY = np.meshgrid(gx, gy)
GZ = dem.ground_elev_xy(GX.ravel(), GY.ravel(), lat0, lon0,
                        natural=True).reshape(GX.shape)

# 公里坐标(原点=包络框西南角)
KX = (GX - x0) / 1000.0; KY = (GY - y0) / 1000.0
kx = (xx - x0) / 1000.0; ky = (yy - y0) / 1000.0
kXA = (XA - x0) / 1000.0; kYA = (YA - y0) / 1000.0

# ---------------- 线形分段 ----------------
eco = dem.eco_mask_xy(xx, yy, lat0, lon0)
cr = CR.detect_crossings(xx, yy, lat0, lon0)
keep = np.interp(cr["s"], sta, eco.astype(float)) <= 0.5 \
    if len(cr["s"]) else np.zeros(0, bool)
iv, L_cross_m = CR.bridge_intervals(cr, keep=keep)
bridge = CR.mask_from_intervals(sta, iv)
normal = ~(eco | bridge)

# ---------------- 地形晕渲 ----------------
# 浅色学术地形色带: 米白 -> 浅卡其 -> 浅棕 -> 灰白(高程)
terrain_cmap = LinearSegmentedColormap.from_list(
    "paper_terrain",
    ["#f4f1ea", "#e4ddc7", "#cfc09a", "#b09b6d", "#8d7550", "#e8e4dd"])
ls = LightSource(azdeg=315, altdeg=42)
rgb = ls.shade(GZ, cmap=terrain_cmap, blend_mode="soft",
               vert_exag=12.0, vmin=GZ.min(), vmax=GZ.max())

fig = plt.figure(figsize=(12.8, 5.2))
ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
ax.set_position([-0.04, -0.22, 0.92, 1.42])   # 3D 轴盒外扩, 裁掉投影留白
ax.plot_surface(KX, KY, GZ, facecolors=rgb, shade=False,
                rstride=1, cstride=1, linewidth=0, antialiased=False,
                zorder=1)

# ---------------- 线形(白描边 halo) ----------------
Z_LIFT = 8.0
HALO = [pe.Stroke(linewidth=3.6, foreground="white", alpha=0.9), pe.Normal()]


def _plot_mask(mask, color, lw, ls_="-", halo=True, z=None, alpha=1.0):
    m = np.asarray(mask, bool)
    idx = np.where(m)[0]
    if len(idx) == 0:
        return
    segs = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
    zline = zz if z is None else z
    for s_ in segs:
        ln, = ax.plot(kx[s_], ky[s_], zline[s_] + Z_LIFT, color=color,
                      lw=lw, ls=ls_, alpha=alpha, zorder=12,
                      solid_capstyle="round")
        if halo:
            ln.set_path_effects(HALO)


# 现状中线(细虚灰, 无 halo, 沉在下层)
ax.plot(kXA, kYA, ZA + Z_LIFT * 0.6, color="#6f6f6f", lw=0.8, ls=(0, (2, 2)),
        alpha=0.8, zorder=11)
_plot_mask(normal, "#b6321f", 1.8)                      # 常规路基: 深红
_plot_mask(bridge, "#e88f1a", 2.6)                      # 交叉桥: 琥珀(加粗)
_plot_mask(eco, "#5b3a8e", 2.2, ls_=(0, (4, 2)))        # 生态隧道: 紫虚线

# ---------------- 坐标与视图 ----------------
Lx = (x1 - x0) / 1000.0; Ly = (y1 - y0) / 1000.0
Z_VIS = 3.6                                    # z 轴视觉高度(图内单位)
ax.set_box_aspect((Lx, Ly, Z_VIS))
zmax = float(np.ceil(GZ.max() / 100.0) * 100.0)
ax.set_zlim(0, zmax)
# 竖向夸张系数 = (视觉高度/实际高差) / (视觉长度/实际长度)
vexag = (Z_VIS / (zmax / 1000.0)) / (Lx / Lx)
ax.view_init(elev=42, azim=-112)

ax.set_xlabel("Easting (km)", labelpad=14)
ax.set_ylabel("Northing (km)", labelpad=-4)
ax.zaxis.set_rotate_label(False)
ax.set_zlabel("Elev. (m)", labelpad=10, rotation=90)
ax.set_yticks(np.arange(0, np.ceil(Ly) + 0.1, 2.0))
ax.set_zticks(np.arange(0, zmax + 1, 200))
ax.tick_params(pad=1, labelsize=8.5)

# 干净的三维面板: 白底、极浅网格、去边框着色
for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
    axis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    axis._axinfo["grid"].update(color=(0, 0, 0, 0.10), linewidth=0.4)
    axis.line.set_color((0.35, 0.35, 0.35, 1.0))
    axis.line.set_linewidth(0.6)

ax.set_title("Fig. C11  Optimized alignment M-C on quasi-natural terrain "
             f"(vertical exaggeration ≈ {vexag:.0f}×)",
             fontsize=11, y=0.88)

# 高程色标(细长, 置右)
sm = ScalarMappable(cmap=terrain_cmap)
sm.set_array(GZ)
cb = fig.colorbar(sm, ax=ax, shrink=0.42, aspect=22, pad=0.005)
cb.set_label("Ground elevation (m)", fontsize=9)
cb.ax.tick_params(labelsize=8)
cb.outline.set_linewidth(0.5)

legend = [
    Line2D([0], [0], color="#b6321f", lw=2.0, label="M-C at-grade"),
    Line2D([0], [0], color="#e88f1a", lw=2.8,
           label=f"Crossing-triggered bridges ({L_cross_m/1000.0:.2f} km, "
                 f"{len(iv)} segments)"),
    Line2D([0], [0], color="#5b3a8e", lw=2.4, ls=(0, (4, 2)),
           label=f"Baiyun eco-tunnel ({mc['L_eco_km']:.2f} km)"),
    Line2D([0], [0], color="#6f6f6f", lw=1.0, ls=(0, (2, 2)),
           label="M-A existing centerline"),
]
ax.legend(handles=legend, loc="upper left", bbox_to_anchor=(0.04, 0.80),
          frameon=False, fontsize=8.6, handlelength=2.4,
          labelspacing=0.55, borderaxespad=0)

for ext in ("png", "pdf"):
    plt.savefig(os.path.join(FIG, f"图C11_最终方案三维mesh.{ext}"),
                bbox_inches="tight", pad_inches=0.1)
print("[图] figures/图C11_最终方案三维mesh (publication style)")
