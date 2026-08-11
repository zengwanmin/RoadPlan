# -*- coding: utf-8 -*-
"""draw_buildings.py — 建筑分布 + 走廊带宽度数据依据 + 全图层叠加。
坐标: 模型局部笛卡尔 (X 东, Y 北, m)。
"""
import numpy as np, logging, math
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.spatial import cKDTree
from shapely.geometry import LineString
from data_loader import load_alignment

R_E = 6378137.0
a = load_alignment()
X, Y, s = a['X'], a['Y'], a['s']
lat0 = math.radians(a['lat'][0]); lon0 = math.radians(a['lon'][0])
def ll2xy(lon, lat):
    return (R_E*math.cos(lat0)*(np.radians(lon)-lon0), R_E*(np.radians(lat)-lat0))

b = np.load('osm/buildings.npz', allow_pickle=False)
BX, BY = ll2xy(b['lon'], b['lat'])
XL = (-20800, 800); YL = (-1400, 3000)
cl = LineString(np.c_[X, Y])
def corridor(W):
    g = cl.buffer(W, cap_style=2, join_style=1)
    gs = g.geoms if g.geom_type == 'MultiPolygon' else [g]
    return [np.array(gm.exterior.coords) for gm in gs]

# ---- 走廊带宽度数据依据: 沿中线逐点, 左右两侧到最近建筑的横向距离 ----
tx = np.gradient(X); ty = np.gradient(Y); tn = np.hypot(tx, ty)+1e-9
nx, ny = -ty/tn, tx/tn                      # 单位左法向
btree = cKDTree(np.c_[BX, BY])
# 沿里程等间隔取样点
step = 50.0
sm = np.arange(0, s[-1], step)
px = np.interp(sm, s, X); py = np.interp(sm, s, Y)
pnx = np.interp(sm, s, nx); pny = np.interp(sm, s, ny)
# 建筑相对各采样点的横向(法向)投影, 分左右取最近
d_left = np.full(len(sm), np.nan); d_right = np.full(len(sm), np.nan)
for i in range(len(sm)):
    idx = btree.query_ball_point([px[i], py[i]], 1200)   # 1.2km 内建筑
    if not idx:
        d_left[i] = d_right[i] = 1200; continue
    dx = BX[idx]-px[i]; dy = BY[idx]-py[i]
    lateral = dx*pnx[i] + dy*pny[i]          # >0 左侧, <0 右侧
    along = np.abs(dx*(-pny[i]) + dy*(pnx[i]))
    near = along < 60                        # 只算该桩号断面附近(±60m)的建筑
    ll = lateral[near & (lateral > 0)]; rr = -lateral[near & (lateral < 0)]
    d_left[i] = ll.min() if len(ll) else 1200
    d_right[i] = rr.min() if len(rr) else 1200

# ============ FIG 6: 建筑分布 + 线位 + 走廊带 ============
fig, ax = plt.subplots(figsize=(14, 7), dpi=140)
hb = ax.hexbin(BX, BY, gridsize=120, cmap='OrRd', mincnt=1, alpha=.9)
cb = fig.colorbar(hb, ax=ax, shrink=.8); cb.set_label('building count per cell')
for W, c in [(2500, '#1f6fd6'), (500, '#111111')]:
    for k, ring in enumerate(corridor(W)):
        ax.plot(ring[:, 0], ring[:, 1], color=c, lw=1.4, ls='--', zorder=3,
                label=('corridor +/-%dm' % W) if k == 0 else None)
ax.plot(X, Y, 'g-', lw=2.2, zorder=4, label='centerline')
ax.set_aspect('equal'); ax.set_xlim(*XL); ax.set_ylim(*YL)
ax.set_xlabel('X East (m)'); ax.set_ylabel('Y North (m)')
ax.set_title('Fig 6  Building distribution (OSM, %d footprints) vs corridor width' % len(BX))
ax.legend(loc='upper center', ncol=3, fontsize=9)
fig.tight_layout(); fig.savefig('../figures/region_fig6_buildings.png', facecolor='white'); plt.close(fig)

# ============ FIG 7: 走廊带可行半宽 (数据依据) ============
fig, ax = plt.subplots(figsize=(14, 5.5), dpi=140)
km = sm/1000.0
ax.fill_between(km, 0, d_left, color='#f4a582', alpha=.6, label='left free width to nearest building')
ax.fill_between(km, 0, -d_right, color='#92c5de', alpha=.6, label='right free width to nearest building')
ax.axhline(500, color='k', ls='--', lw=1); ax.axhline(-500, color='k', ls='--', lw=1)
ax.axhline(200, color='r', ls=':', lw=1); ax.axhline(-200, color='r', ls=':', lw=1)
ax.text(0.2, 520, '+/-500 m (our wide setting)', fontsize=9)
ax.text(0.2, 210, '+/-200 m (urban-rebuild)', fontsize=9, color='r')
ax.set_xlabel('mileage (km)'); ax.set_ylabel('lateral free width to nearest building (m)')
ax.set_title('Fig 7  Data-driven corridor half-width: lateral clearance to nearest building along the route')
ax.set_ylim(-1300, 1300); ax.legend(loc='upper right', fontsize=9); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig('../figures/region_fig7_corridor_width.png', facecolor='white'); plt.close(fig)

# 统计: 有多少比例的里程 左右两侧都 >= 阈值
for W in (200, 500):
    ok = np.mean((d_left >= W) & (d_right >= W))*100
    print('走廊 +/-%dm 两侧均无建筑冲突的里程占比: %.1f%%' % (W, ok))

# ============ FIG 8: 全图层叠加 ============
d = np.load('dem_xwide_z14.npz'); E = d['elev']; Z = int(d['z']); x0 = int(d['x0']); y0 = int(d['y0'])
H, Wd = E.shape; n = 2**Z
jj = np.arange(Wd); ii = np.arange(H)
lon_px = (x0+(jj+0.5)/256.0)/n*360.0-180.0
lat_rd = np.arctan(np.sinh(np.pi*(1-2*(y0+(ii+0.5)/256.0)/n)))
GLON, GLAT = np.meshgrid(np.radians(lon_px), lat_rd)
DX = R_E*math.cos(lat0)*(GLON-lon0); DY = R_E*(GLAT-lat0)
Emask = np.where(E < -100, np.nan, E)
o = np.load('osm/obstacles.npz', allow_pickle=False)
OX, OY = ll2xy(o['lines_lon'], o['lines_lat']); OFF = o['offsets']; KND = o['kind']
COL = {'road': '#777777', 'rail': '#7b3fa0', 'water': '#2b8cbe'}

fig, ax = plt.subplots(figsize=(15, 7.6), dpi=140)
ax.pcolormesh(DX, DY, Emask, cmap='terrain', shading='auto', alpha=.62, zorder=0)
ax.scatter(BX, BY, s=2, color='#8b0000', alpha=.35, zorder=1)
for i in range(len(KND)):
    sl = slice(OFF[i], OFF[i+1])
    ax.plot(OX[sl], OY[sl], color=COL[KND[i]], lw=0.5, alpha=.55, zorder=2)
for k, ring in enumerate(corridor(500)):
    ax.plot(ring[:, 0], ring[:, 1], color='#1f6fd6', lw=1.6, ls='--', zorder=3,
            label='corridor +/-500 m' if k == 0 else None)
ax.plot(X, Y, 'r-', lw=2.2, zorder=4, label='centerline')
ax.set_aspect('equal'); ax.set_xlim(*XL); ax.set_ylim(*YL)
ax.set_xlabel('X East (m)'); ax.set_ylabel('Y North (m)')
ax.set_title('Fig 8  Full overlay: DEM + buildings + transport network + centerline + corridor')
h = [Line2D([0],[0],marker='o',color='w',markerfacecolor='#8b0000',ms=6,label='building'),
     Line2D([0],[0],color='#777777',lw=2,label='road'),
     Line2D([0],[0],color='#7b3fa0',lw=2,label='rail'),
     Line2D([0],[0],color='#2b8cbe',lw=2,label='water'),
     Line2D([0],[0],color='r',lw=2,label='centerline'),
     Line2D([0],[0],color='#1f6fd6',lw=1.6,ls='--',label='corridor +/-500 m')]
ax.legend(handles=h, loc='upper center', ncol=3, fontsize=9)
fig.tight_layout(); fig.savefig('../figures/region_fig8_alllayers.png', facecolor='white'); plt.close(fig)
print('done fig6/fig7/fig8')
