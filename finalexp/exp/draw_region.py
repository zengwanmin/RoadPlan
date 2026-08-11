# -*- coding: utf-8 -*-
"""draw_region.py — 建设区域真实可视化(3 图):
  fig1: 实测平面线形在笛卡尔系的真实样子 + 走廊带可行域(±500/±2500)
  fig2: DEM 勾勒的宽泛平面地形(整片) + 线位叠加
  fig3: 综合建设区域(DEM 底 + 线位 + 走廊带 + 桥隧标注)
坐标: 模型局部笛卡尔 (X 东向, Y 北向, 单位 m), 原点=实测起点。
"""
import numpy as np, logging, math, re
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from data_loader import load_alignment

R_E = 6378137.0
a = load_alignment()
X, Y, s = a['X'], a['Y'], a['s']
lat0, lon0 = math.radians(a['lat'][0]), math.radians(a['lon'][0])

# ---- 单位左法向(用于走廊带边界) ----
tx = np.gradient(X); ty = np.gradient(Y)
tn = np.hypot(tx, ty) + 1e-9
nx, ny = -ty/tn, tx/tn

from shapely.geometry import LineString
from shapely.ops import unary_union
_cl = LineString(np.c_[X, Y])
def corridor_poly(W):
    """真实走廊 = 中线的 W 缓冲区(不自交), 返回外边界坐标数组列表。"""
    g = _cl.buffer(W, cap_style=2, join_style=1)
    geoms = g.geoms if g.geom_type == 'MultiPolygon' else [g]
    return [np.array(gm.exterior.coords) for gm in geoms]

# ---- DEM -> 局部笛卡尔网格 ----
d = np.load('dem_xwide_z14.npz')
E = d['elev']; Z = int(d['z']); x0 = int(d['x0']); y0 = int(d['y0'])
H, Wd = E.shape; n = 2**Z
# 像素中心 -> 经纬度 -> X,Y
jj = np.arange(Wd); ii = np.arange(H)
lon_px = (x0 + (jj+0.5)/256.0)/n*360.0 - 180.0
lat_rad = np.arctan(np.sinh(np.pi*(1 - 2*(y0 + (ii+0.5)/256.0)/n)))
LON, LAT = np.meshgrid(np.radians(lon_px), lat_rad)
DX = R_E*math.cos(lat0)*(LON - lon0)
DY = R_E*(LAT - lat0)
Emask = np.where(E < -100, np.nan, E)

# ---- 桥隧: 桩号(米) -> 平面坐标(按 s 里程插值) ----
df = pd.read_excel('/root/roadplan/RoadPlan_remote/数据/北环高速现状桥梁隧道统计.xlsx')
def parse_k(t):
    m = re.findall(r'K(\d+)\+(\d+)', str(t))
    return [int(a)*1000+int(b) for a, b in m]
segs = []
for _, r in df.iloc[1:9].iterrows():
    ks = parse_k(r.iloc[0])
    if len(ks) == 2:
        segs.append((r.iloc[1], ks[0], ks[1]))
def seg_xy(m0, m1):
    mm = np.linspace(m0, m1, 40)
    return np.interp(mm, s, X), np.interp(mm, s, Y)

XL = (-20800, 800); YL = (-1400, 3000)

# ================= FIG 1: 实测平面线形 + 走廊带 =================
fig, ax = plt.subplots(figsize=(13, 6.5), dpi=140)
for W, c, lab in [(2500,'#cfe8ff','corridor +/-2500 m'),(500,'#ffe0a3','corridor +/-500 m')]:
    for k, ring in enumerate(corridor_poly(W)):
        ax.fill(ring[:,0], ring[:,1], color=c, alpha=.75,
                label=lab if k==0 else None, zorder=1)
ax.plot(X, Y, 'k-', lw=2.4, label='measured centerline (existing road)', zorder=3)
ax.plot(X[0], Y[0], 'o', ms=13, color='green', zorder=4); ax.annotate('A start (K0)', (X[0]+150,Y[0]-120), color='green', fontsize=11)
ax.plot(X[-1], Y[-1], 's', ms=12, color='red', zorder=4); ax.annotate('B end (K22.5)', (X[-1]-2600,Y[-1]+60), color='red', fontsize=11)
ax.set_aspect('equal'); ax.set_xlim(*XL); ax.set_ylim(*YL)
ax.set_xlabel('X  East (m)'); ax.set_ylabel('Y  North (m)')
ax.set_title('Fig 1  Measured plan alignment in Cartesian frame  +  plan feasible region (corridor)')
ax.legend(loc='upper center', ncol=2, fontsize=10); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig('../figures/region_fig1_plan.png', facecolor='white'); plt.close(fig)

# ================= FIG 2: DEM 宽泛平面地形 =================
fig, ax = plt.subplots(figsize=(13, 6.5), dpi=140)
pc = ax.pcolormesh(DX, DY, Emask, cmap='terrain', shading='auto')
cb = fig.colorbar(pc, ax=ax, shrink=.85); cb.set_label('ground elevation (m)')
ax.plot(X, Y, 'r-', lw=2, label='measured centerline')
ax.set_aspect('equal'); ax.set_xlim(*XL); ax.set_ylim(*YL)
ax.set_xlabel('X  East (m)'); ax.set_ylabel('Y  North (m)')
ax.set_title('Fig 2  Real terrain (DEM) over the wide plan region  (AWS Terrain Tiles z14, ~8.8 m)')
ax.legend(loc='upper right', fontsize=10)
fig.tight_layout(); fig.savefig('../figures/region_fig2_dem.png', facecolor='white'); plt.close(fig)

# ================= FIG 3: 综合建设区域 =================
fig, ax = plt.subplots(figsize=(13.5, 7), dpi=140)
pc = ax.pcolormesh(DX, DY, Emask, cmap='terrain', shading='auto', alpha=.85)
cb = fig.colorbar(pc, ax=ax, shrink=.85); cb.set_label('ground elevation (m)')
for k, ring in enumerate(corridor_poly(500)):
    ax.plot(ring[:,0], ring[:,1], color='#1f6fd6', lw=1.6, ls='--',
            label='corridor +/-500 m' if k==0 else None, zorder=2)
ax.plot(X, Y, 'k-', lw=2, label='centerline', zorder=3)
bshown=tshown=False
for typ, m0, m1 in segs:
    sx, sy = seg_xy(m0, m1)
    if typ == '桥梁':
        ax.plot(sx, sy, lw=6, color='#e6550d', solid_capstyle='round', zorder=4,
                label=None if bshown else 'bridge (existing)'); bshown=True
    else:
        ax.plot(sx, sy, lw=7, color='#6a1b9a', solid_capstyle='round', zorder=4,
                label=None if tshown else 'tunnel (existing)'); tshown=True
ax.plot(X[0], Y[0], 'o', ms=12, color='green', zorder=5)
ax.plot(X[-1], Y[-1], 's', ms=11, color='red', zorder=5)
ax.set_aspect('equal'); ax.set_xlim(*XL); ax.set_ylim(*YL)
ax.set_xlabel('X  East (m)'); ax.set_ylabel('Y  North (m)')
ax.set_title('Fig 3  Construction region overview: terrain + centerline + corridor + existing bridges/tunnel')
ax.legend(loc='upper center', ncol=3, fontsize=10)
fig.tight_layout(); fig.savefig('../figures/region_fig3_overview.png', facecolor='white'); plt.close(fig)

print('done: region_fig1_plan / region_fig2_dem / region_fig3_overview')
print('bridges/tunnel segments:', [(t,m0,m1) for t,m0,m1 in segs])
