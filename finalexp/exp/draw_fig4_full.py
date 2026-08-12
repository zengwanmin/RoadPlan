# -*- coding: utf-8 -*-
"""
draw_fig4_full.py — 完备 OSM 建筑集在二维模型坐标下的分布图(与 region_fig4_osm.png 同款式)

与 region_fig4_osm.png 的关系: 那张图画的是走廊带内的 OSM 交通网络(道路/铁路/水系),
本图沿用同一坐标系(模型局部笛卡尔, X 东 / Y 北, 原点=实测线位起点)、同一套配色与
图例款式, 把图层换成【完备建筑集的真实轮廓多边形】。

数据: osm/buildings_full.npz — 22590 个建筑, 已与 Overpass `out count` 逐项核对一致
      (way 22414/22414, relation 177/177), 171039 个顶点, 总占地 24.41 km²。

绘图范围取建筑数据自身的覆盖范围(即抓取 bbox 对应的 XY 范围), 而不沿用 fig4 的裁剪窗口
—— 本图的用途正是展示建筑集的完备性, 裁掉窗口外的建筑会与该用途矛盾。
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
from matplotlib.patches import Patch

from data_loader import load_alignment

R_E = 6378137.0

a = load_alignment()
X, Y = a['X'], a['Y']
lat0 = math.radians(a['lat'][0])
lon0 = math.radians(a['lon'][0])


def ll2xy(lon, lat):
    return (R_E * math.cos(lat0) * (np.radians(lon) - lon0),
            R_E * (np.radians(lat) - lat0))


d = np.load('osm/buildings_full.npz', allow_pickle=False)
PX, PY = ll2xy(d['poly_lon'], d['poly_lat'])
OFF, RING = d['poly_off'], d['poly_ring']
AREA = d['b_area_m2']
NB = len(d['b_id'])

# 只填充外环(内环即天井)
outer = [np.c_[PX[OFF[k]:OFF[k + 1]], PY[OFF[k]:OFF[k + 1]]]
         for k in range(len(RING)) if RING[k] == 'outer']
print(f"[数据] 建筑 {NB} 个, 外环 {len(outer)} 个, 顶点 {len(PX)} 个, "
      f"总占地 {AREA.sum()/1e6:.2f} km²")

mg = 300.0
XL = (PX.min() - mg, PX.max() + mg)
YL = (PY.min() - mg, PY.max() + mg)
print(f"[范围] X {XL[0]:.0f}~{XL[1]:.0f} m, Y {YL[0]:.0f}~{YL[1]:.0f} m")

span_x = XL[1] - XL[0]
span_y = YL[1] - YL[0]
fig_w = 15.0
fig_h = max(4.5, min(11.0, fig_w * span_y / span_x + 1.6))

fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=140)
ax.set_rasterization_zorder(3)
ax.add_collection(PolyCollection(outer, facecolors='#8b0000', edgecolors='none',
                                 alpha=0.85, zorder=1))
ax.plot(X, Y, 'r-', lw=2.4, zorder=4, label='highway centerline')
ax.plot(X[0], Y[0], 'o', ms=12, color='green', zorder=5)
ax.plot(X[-1], Y[-1], 's', ms=11, color='red', zorder=5)

ax.set_aspect('equal')
ax.set_xlim(*XL)
ax.set_ylim(*YL)
ax.set_xlabel('X East (m)')
ax.set_ylabel('Y North (m)')
ax.set_title('Fig 4-full  Complete OSM building footprints in corridor region '
             f'({NB} buildings, {AREA.sum()/1e6:.1f} km$^2$ built-up area)'
             '  — verified against Overpass out count')
leg = [Patch(facecolor='#8b0000', edgecolor='none',
             label=f'building footprint ({NB} features, outer rings)'),
       Line2D([0], [0], color='red', lw=2, label='highway centerline'),
       Line2D([0], [0], marker='o', color='w', markerfacecolor='green', ms=10,
              label='route start'),
       Line2D([0], [0], marker='s', color='w', markerfacecolor='red', ms=10,
              label='route end')]
ax.legend(handles=leg, loc='upper center', ncol=2, fontsize=9)
fig.tight_layout()
fig.savefig('../figures/fig4_full.png', facecolor='white')
plt.close(fig)
print('saved ../figures/fig4_full.png')
