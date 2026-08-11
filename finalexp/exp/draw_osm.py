# -*- coding: utf-8 -*-
"""draw_osm.py — OSM 交通网络可视化 + 四层综合图(DEM/交通网/线位/走廊带/高架触发点)。
坐标: 模型局部笛卡尔 (X 东, Y 北, m), 原点=实测起点。
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

# ---- OSM 障碍物折线 ----
o = np.load('osm/obstacles.npz', allow_pickle=False)
LON, LAT = o['lines_lon'], o['lines_lat']
OFF = o['offsets']; KIND = o['kind']
OX, OY = ll2xy(LON, LAT)
COL = {'road': '#888888', 'rail': '#7b3fa0', 'water': '#2b8cbe'}
LW = {'road': 0.7, 'rail': 1.1, 'water': 1.0}

def draw_network(ax, alpha=1.0):
    for i in range(len(KIND)):
        sl = slice(OFF[i], OFF[i+1])
        ax.plot(OX[sl], OY[sl], color=COL[KIND[i]], lw=LW[KIND[i]], alpha=alpha, zorder=2)

# ---- 走廊带(shapely buffer) ----
cl = LineString(np.c_[X, Y])
def corridor(W):
    g = cl.buffer(W, cap_style=2, join_style=1)
    gs = g.geoms if g.geom_type == 'MultiPolygon' else [g]
    return [np.array(gm.exterior.coords) for gm in gs]

# ---- 高架触发点: 障碍物顶点落入走廊带 且 距中线 < 阈值 ----
poly500 = cl.buffer(500)
tree = cKDTree(np.c_[X, Y])
# 交叉点: 障碍物顶点距中线 < 40 m (视为线位需跨越处), 逐顶点判定
dist, _ = tree.query(np.c_[OX, OY])
cross = (dist < 40)
cx, cy = OX[cross], OY[cross]

XL = (-20800, 800); YL = (-1400, 3000)

# ============ FIG A: 纯交通网络 ============
fig, ax = plt.subplots(figsize=(13.5, 6.8), dpi=140)
draw_network(ax)
ax.plot(X, Y, 'r-', lw=2.4, zorder=4, label='highway centerline')
ax.plot(X[0], Y[0], 'o', ms=12, color='green', zorder=5)
ax.plot(X[-1], Y[-1], 's', ms=11, color='red', zorder=5)
ax.set_aspect('equal'); ax.set_xlim(*XL); ax.set_ylim(*YL)
ax.set_xlabel('X East (m)'); ax.set_ylabel('Y North (m)')
ax.set_title('Fig 4  OSM transport network in corridor (roads / rail / water)  — grade-separation obstacles')
leg = [Line2D([0],[0],color='#888888',lw=2,label='road (motorway/trunk/primary/secondary)'),
       Line2D([0],[0],color='#7b3fa0',lw=2,label='railway'),
       Line2D([0],[0],color='#2b8cbe',lw=2,label='river/canal'),
       Line2D([0],[0],color='red',lw=2,label='highway centerline')]
ax.legend(handles=leg, loc='upper center', ncol=2, fontsize=9)
fig.tight_layout(); fig.savefig('../figures/region_fig4_osm.png', facecolor='white'); plt.close(fig)

# ============ FIG B: 四层综合 ============
d = np.load('dem_xwide_z14.npz'); E = d['elev']; Z = int(d['z']); x0 = int(d['x0']); y0 = int(d['y0'])
H, Wd = E.shape; n = 2**Z
jj = np.arange(Wd); ii = np.arange(H)
lon_px = (x0 + (jj+0.5)/256.0)/n*360.0 - 180.0
lat_rad = np.arctan(np.sinh(np.pi*(1 - 2*(y0 + (ii+0.5)/256.0)/n)))
GLON, GLAT = np.meshgrid(np.radians(lon_px), lat_rad)
DX = R_E*math.cos(lat0)*(GLON - lon0); DY = R_E*(GLAT - lat0)
Emask = np.where(E < -100, np.nan, E)

fig, ax = plt.subplots(figsize=(14.5, 7.4), dpi=140)
pc = ax.pcolormesh(DX, DY, Emask, cmap='terrain', shading='auto', alpha=.8, zorder=0)
cb = fig.colorbar(pc, ax=ax, shrink=.82); cb.set_label('ground elevation (m)')
draw_network(ax, alpha=.55)
for k, ring in enumerate(corridor(500)):
    ax.plot(ring[:,0], ring[:,1], color='#1f6fd6', lw=1.6, ls='--', zorder=3,
            label='corridor +/-500 m' if k == 0 else None)
ax.plot(X, Y, 'r-', lw=2.2, zorder=4, label='highway centerline')
ax.scatter(cx, cy, s=26, marker='v', color='#d62728', edgecolor='k', lw=.4,
           zorder=5, label='grade-separation trigger (obstacle crossing)')
ax.plot(X[0], Y[0], 'o', ms=12, color='green', zorder=6)
ax.plot(X[-1], Y[-1], 's', ms=11, color='red', zorder=6)
ax.set_aspect('equal'); ax.set_xlim(*XL); ax.set_ylim(*YL)
ax.set_xlabel('X East (m)'); ax.set_ylabel('Y North (m)')
ax.set_title('Fig 5  Integrated construction map: DEM + transport network + centerline + corridor + viaduct triggers')
h1 = [Line2D([0],[0],color='#888888',lw=2,label='road'),
      Line2D([0],[0],color='#7b3fa0',lw=2,label='rail'),
      Line2D([0],[0],color='#2b8cbe',lw=2,label='water'),
      Line2D([0],[0],color='r',lw=2,label='centerline'),
      Line2D([0],[0],color='#1f6fd6',lw=1.6,ls='--',label='corridor +/-500 m'),
      Line2D([0],[0],marker='v',color='w',markerfacecolor='#d62728',markeredgecolor='k',ms=9,label='viaduct trigger')]
ax.legend(handles=h1, loc='upper center', ncol=3, fontsize=9)
fig.tight_layout(); fig.savefig('../figures/region_fig5_integrated.png', facecolor='white'); plt.close(fig)

print('done fig4_osm / fig5_integrated ;  交叉高架触发点数:', int(cross.sum()))
