# -*- coding: utf-8 -*-
"""
draw_buildings_full.py — 用完整建筑轮廓重做建筑分布图与走廊带宽度依据图

【与旧版 draw_buildings.py 的关键差别】
  旧版用 12789 个建筑【质心点】(且 bbox 偏窄、缺多重多边形), 本版用 22590 个建筑的
  【完整轮廓多边形】(171039 个顶点)。这对"走廊带可行半宽"是实质性差别: 横向净距
  应量到建筑【边缘】而非质心, 大型建筑(最大占地 14.5 万 m²)的边缘比质心近上百米,
  用质心会系统性高估可用宽度。

输出:
  region_fig6b_buildings_footprint.png  建筑轮廓分布 + 线位 + 走廊带
  region_fig7b_corridor_width_full.png  按轮廓边缘量测的走廊带可行半宽
© OpenStreetMap contributors, ODbL v1.0。
"""
import math
import numpy as np
import logging
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from scipy.spatial import cKDTree
from shapely.geometry import LineString

from data_loader import load_alignment

R_E = 6378137.0
NPZ = 'osm/buildings_full.npz'

a = load_alignment()
X, Y, s = a['X'], a['Y'], a['s']
lat0 = math.radians(a['lat'][0])
lon0 = math.radians(a['lon'][0])


def ll2xy(lon, lat):
    return (R_E * math.cos(lat0) * (np.radians(lon) - lon0),
            R_E * (np.radians(lat) - lat0))


d = np.load(NPZ, allow_pickle=False)
PX, PY = ll2xy(d['poly_lon'], d['poly_lat'])
OFF = d['poly_off']
RING = d['poly_ring']
AREA = d['b_area_m2']
print(f"[数据] 建筑 {len(AREA)} 个, 环 {len(RING)} 个, 顶点 {len(PX)} 个, "
      f"总占地 {AREA.sum()/1e6:.2f} km²")

# 只用外环画填充多边形(内环即天井, 对净距计算无影响)
polys = [np.c_[PX[OFF[k]:OFF[k + 1]], PY[OFF[k]:OFF[k + 1]]]
         for k in range(len(RING)) if RING[k] == 'outer']

cl = LineString(np.c_[X, Y])


def corridor(Wm):
    g = cl.buffer(Wm, cap_style=2, join_style=1)
    gs = g.geoms if g.geom_type == 'MultiPolygon' else [g]
    return [np.array(gm.exterior.coords) for gm in gs]


# ---------- 走廊带可行半宽: 量到建筑轮廓【顶点】而非质心 ----------
tx = np.gradient(X); ty = np.gradient(Y)
tn = np.hypot(tx, ty) + 1e-9
nx, ny = -ty / tn, tx / tn                      # 单位左法向
step = 25.0
sm = np.arange(0, s[-1], step)
px = np.interp(sm, s, X); py = np.interp(sm, s, Y)
pnx = np.interp(sm, s, nx); pny = np.interp(sm, s, ny)

tree = cKDTree(np.c_[PX, PY])
CAP = 1200.0
d_left = np.full(len(sm), CAP)
d_right = np.full(len(sm), CAP)
for i in range(len(sm)):
    idx = tree.query_ball_point([px[i], py[i]], CAP)
    if not idx:
        continue
    idx = np.asarray(idx)
    dx = PX[idx] - px[i]; dy = PY[idx] - py[i]
    lateral = dx * pnx[i] + dy * pny[i]         # >0 左, <0 右
    along = np.abs(dx * (-pny[i]) + dy * pnx[i])
    near = along < 30.0                         # 该桩号断面附近 ±30 m
    ll = lateral[near & (lateral > 0)]
    rr = -lateral[near & (lateral < 0)]
    if len(ll):
        d_left[i] = ll.min()
    if len(rr):
        d_right[i] = rr.min()

XL = (X.min() - 1500, X.max() + 1500)
YL = (Y.min() - 1800, Y.max() + 1800)

# ============ FIG 6b: 建筑轮廓分布 ============
fig, ax = plt.subplots(figsize=(15.2, 5.6), dpi=140)
ax.set_rasterization_zorder(3)
ax.add_collection(PolyCollection(polys, facecolors='#8b0000', edgecolors='none',
                                 alpha=0.75, zorder=1))
for Wm, c in [(500, '#1f6fd6'), (200, '#111111')]:
    for k, ring in enumerate(corridor(Wm)):
        ax.plot(ring[:, 0], ring[:, 1], color=c, lw=1.3, ls='--', zorder=4,
                label=f'corridor +/-{Wm} m' if k == 0 else None)
ax.plot(X, Y, color='#00a000', lw=2.0, zorder=5, label='existing centerline')
ax.set_aspect('equal'); ax.set_xlim(*XL); ax.set_ylim(*YL)
ax.set_xlabel('Easting X (m)'); ax.set_ylabel('Northing Y (m)')
ax.set_title('Fig 6b  Complete OSM building footprints '
             f'({len(AREA)} buildings, {AREA.sum()/1e6:.1f} km$^2$ built area) '
             'vs corridor width')
ax.legend(loc='upper right', fontsize=9)
fig.tight_layout()
fig.savefig('../figures/region_fig6b_buildings_footprint.png', facecolor='white')
plt.close(fig)
print('saved region_fig6b_buildings_footprint.png')

# ============ FIG 7b: 走廊带可行半宽 ============
fig, ax = plt.subplots(figsize=(15.2, 5.2), dpi=140)
km = sm / 1000.0
ax.fill_between(km, 0, d_left, color='#f4a582', alpha=.75,
                label='left clearance to nearest building edge')
ax.fill_between(km, 0, -d_right, color='#92c5de', alpha=.75,
                label='right clearance to nearest building edge')
for v, c, ls, t in [(500, 'k', '--', '+/-500 m'), (200, 'r', ':', '+/-200 m')]:
    ax.axhline(v, color=c, ls=ls, lw=1); ax.axhline(-v, color=c, ls=ls, lw=1)
    ax.text(0.15, v + 25, t, fontsize=9, color=c)
ax.set_xlabel('mileage (km)')
ax.set_ylabel('lateral clearance to nearest building edge (m)')
ax.set_title('Fig 7b  Data-driven corridor half-width, measured to building '
             'FOOTPRINT EDGES (complete OSM set)')
ax.set_ylim(-1300, 1300); ax.legend(loc='upper right', fontsize=9)
ax.grid(alpha=.3)
fig.tight_layout()
fig.savefig('../figures/region_fig7b_corridor_width_full.png', facecolor='white')
plt.close(fig)
print('saved region_fig7b_corridor_width_full.png')

print('\n[走廊带宽度统计] 两侧均无建筑冲突的里程占比 (量到轮廓边缘):')
for Wm in (100, 200, 300, 500, 800):
    ok = np.mean((d_left >= Wm) & (d_right >= Wm)) * 100
    print(f'  +/-{Wm:4d} m : {ok:5.1f}%')
print(f'[净距] 左侧中位 {np.median(d_left):.0f} m, 右侧中位 {np.median(d_right):.0f} m; '
      f'最紧断面 {min(d_left.min(), d_right.min()):.0f} m')
