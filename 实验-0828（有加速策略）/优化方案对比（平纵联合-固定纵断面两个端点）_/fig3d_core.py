# -*- coding: utf-8 -*-
"""
fig3d_core.py — 最终方案(M-C)三维地形图的共享渲染核心。

各版本薄脚本(make_fig_C11_vA..vE.py)只提供一个配置 dict, 调 render(cfg) 出图。
数据与几何(DEM mesh / M-C 线形 / 桥隧分段)在本模块加载一次, 版本间完全一致,
差异仅在: 地形配色、光照、视角、线形配色、明暗主题、显示范围。
输出目录: figures_三维版本/
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
OUT = os.path.join(HERE, "figures_三维版本")
os.makedirs(OUT, exist_ok=True)

from data_loader import load_alignment
import dem
import crossings as CR

# ---------------- 数据(模块级加载一次) ----------------
with open(os.path.join(HERE, "results", "joint_results.json"),
          encoding="utf-8") as f:
    _d = json.load(f)
MC = _d["M_C"]
XX = np.array(MC["plane_x"]); YY = np.array(MC["plane_y"])
ZZ = np.array(MC["design_z"]); STA = np.array(MC["sta"])

_align = load_alignment()
LAT0, LON0 = float(_align["lat"][0]), float(_align["lon"][0])
XA, YA, ZA = _align["X"], _align["Y"], _align["ground_z"]

PAD = 700.0
X0 = min(XX.min(), XA.min()) - PAD; X1 = max(XX.max(), XA.max()) + PAD
Y0 = min(YY.min(), YA.min()) - PAD; Y1 = max(YY.max(), YA.max()) + PAD
_gx = np.linspace(X0, X1, 480); _gy = np.linspace(Y0, Y1, 200)
GX, GY = np.meshgrid(_gx, _gy)
GZ = dem.ground_elev_xy(GX.ravel(), GY.ravel(), LAT0, LON0,
                        natural=True).reshape(GX.shape)

ECO = dem.eco_mask_xy(XX, YY, LAT0, LON0)
_cr = CR.detect_crossings(XX, YY, LAT0, LON0)
_keep = np.interp(_cr["s"], STA, ECO.astype(float)) <= 0.5 \
    if len(_cr["s"]) else np.zeros(0, bool)
IV, L_CROSS_M = CR.bridge_intervals(_cr, keep=_keep)
BRIDGE = CR.mask_from_intervals(STA, IV)
NORMAL = ~(ECO | BRIDGE)

plt.rcParams.update({
    "figure.dpi": 200, "font.size": 9.5, "font.family": "DejaVu Sans",
    "axes.linewidth": 0.6, "mathtext.default": "regular",
})


def render(cfg):
    """按配置渲染并保存一版三维图。cfg 关键项见各版本脚本。"""
    dark = cfg.get("dark", False)
    fg = "#e8e8e8" if dark else "#222222"
    grid_a = 0.16 if dark else 0.10
    halo_c = ("#101418" if dark else "white")

    # 显示范围(米, 世界坐标); None=全线
    ext = cfg.get("extent")
    if ext:
        ex0, ex1 = ext
        col = (GX[0] >= ex0) & (GX[0] <= ex1)
        KXs = GX[:, col]; KYs = GY[:, col]; GZs = GZ[:, col]
        line_in = (XX >= ex0) & (XX <= ex1)
    else:
        KXs, KYs, GZs = GX, GY, GZ
        line_in = np.ones(len(XX), bool)
    kx0 = KXs.min(); ky0 = KYs.min()
    KX = (KXs - kx0) / 1000.0; KY = (KYs - ky0) / 1000.0

    cmap = LinearSegmentedColormap.from_list(cfg["name"], cfg["terrain"])
    ls = LightSource(azdeg=cfg.get("light_az", 315),
                     altdeg=cfg.get("light_alt", 42))
    rgb = ls.shade(GZs, cmap=cmap, blend_mode=cfg.get("blend", "soft"),
                   vert_exag=cfg.get("shade_exag", 12.0),
                   vmin=GZs.min(), vmax=GZs.max())

    fig = plt.figure(figsize=cfg.get("figsize", (12.8, 5.2)))
    if dark:
        fig.patch.set_facecolor("#14171c")
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
    ax.set_position(cfg.get("axpos", [-0.04, -0.22, 0.92, 1.42]))
    if dark:
        ax.set_facecolor("#14171c")
    ax.plot_surface(KX, KY, GZs, facecolors=rgb, shade=False,
                    rstride=1, cstride=1, linewidth=0, antialiased=False,
                    zorder=1)

    Z_LIFT = 8.0
    HALO = [pe.Stroke(linewidth=3.8, foreground=halo_c, alpha=0.9),
            pe.Normal()]
    kxl = (XX - kx0) / 1000.0; kyl = (YY - ky0) / 1000.0

    def _pl(mask, color, lw, ls_="-"):
        m = np.asarray(mask, bool) & line_in
        idx = np.where(m)[0]
        if len(idx) == 0:
            return
        for s_ in np.split(idx, np.where(np.diff(idx) > 1)[0] + 1):
            ln, = ax.plot(kxl[s_], kyl[s_], ZZ[s_] + Z_LIFT, color=color,
                          lw=lw, ls=ls_, zorder=12, solid_capstyle="round")
            ln.set_path_effects(HALO)

    ma_in = (XA >= (ext[0] if ext else X0)) & (XA <= (ext[1] if ext else X1))
    ax.plot((XA[ma_in] - kx0) / 1000.0, (YA[ma_in] - ky0) / 1000.0,
            ZA[ma_in] + Z_LIFT * 0.6, color=cfg.get("ma_color", "#6f6f6f"),
            lw=0.8, ls=(0, (2, 2)), alpha=0.85, zorder=11)
    c_road, c_br, c_tu = cfg["line_colors"]
    _pl(NORMAL, c_road, 1.8)
    _pl(BRIDGE, c_br, 2.6)
    _pl(ECO, c_tu, 2.2, ls_=(0, (4, 2)))

    Lx = KX.max(); Ly = KY.max()
    Z_VIS = cfg.get("z_vis", 3.6)
    ax.set_box_aspect((Lx, Ly, Z_VIS))
    zmax = float(np.ceil(GZs.max() / 100.0) * 100.0)
    ax.set_zlim(0, zmax)
    vexag = (Z_VIS / (zmax / 1000.0))
    ax.view_init(elev=cfg.get("elev", 42), azim=cfg.get("azim", -112))

    ax.set_xlabel("Easting (km)", labelpad=14, color=fg)
    ax.set_ylabel("Northing (km)", labelpad=-4, color=fg)
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel("Elev. (m)", labelpad=10, rotation=90, color=fg)
    ax.set_yticks(np.arange(0, np.ceil(Ly) + 0.1, 2.0))
    ax.set_zticks(np.arange(0, zmax + 1, 200))
    ax.tick_params(pad=1, labelsize=8.5, colors=fg)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((1, 1, 1, 0))
        axis._axinfo["grid"].update(color=(fg, grid_a)
                                    if False else
                                    ((1, 1, 1, grid_a) if dark
                                     else (0, 0, 0, grid_a)),
                                    linewidth=0.4)
        axis.line.set_color(fg)
        axis.line.set_linewidth(0.6)

    ax.set_title(f"Fig. C11 ({cfg['tag']})  Optimized alignment M-C on "
                 f"quasi-natural terrain (vert. exag. ≈ {vexag:.0f}×)",
                 fontsize=11, y=cfg.get("title_y", 0.88), color=fg)

    sm = ScalarMappable(cmap=cmap); sm.set_array(GZs)
    cb = fig.colorbar(sm, ax=ax, shrink=0.42, aspect=22, pad=0.005)
    cb.set_label("Ground elevation (m)", fontsize=9, color=fg)
    cb.ax.tick_params(labelsize=8, colors=fg)
    cb.outline.set_linewidth(0.5); cb.outline.set_edgecolor(fg)

    legend = [
        Line2D([0], [0], color=c_road, lw=2.0, label="M-C at-grade"),
        Line2D([0], [0], color=c_br, lw=2.8,
               label=f"Crossing-triggered bridges ({L_CROSS_M/1000:.2f} km, "
                     f"{len(IV)} seg.)"),
        Line2D([0], [0], color=c_tu, lw=2.4, ls=(0, (4, 2)),
               label=f"Baiyun eco-tunnel ({MC['L_eco_km']:.2f} km)"),
        Line2D([0], [0], color=cfg.get("ma_color", "#6f6f6f"), lw=1.0,
               ls=(0, (2, 2)), label="M-A existing centerline"),
    ]
    leg = ax.legend(handles=legend, loc="upper left",
                    bbox_to_anchor=cfg.get("legend_at", (0.04, 0.80)),
                    frameon=False, fontsize=8.6, handlelength=2.4,
                    labelspacing=0.55, borderaxespad=0)
    for t in leg.get_texts():
        t.set_color(fg)

    fn = os.path.join(OUT, f"图C11_{cfg['tag']}_{cfg['name']}")
    for ext_ in ("png", "pdf"):
        plt.savefig(f"{fn}.{ext_}", bbox_inches="tight", pad_inches=0.1,
                    facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[图] figures_三维版本/图C11_{cfg['tag']}_{cfg['name']}")
