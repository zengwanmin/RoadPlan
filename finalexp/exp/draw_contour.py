# -*- coding: utf-8 -*-
"""draw_contour.py — 沿平面线形的地面高程趋势(纵断面剖面), 0 m 为基准线。"""
import numpy as np, logging, math
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from data_loader import load_alignment
import dem

a = load_alignment()
lat0d, lon0d = a['lat'][0], a['lon'][0]
s_km = a['s'] / 1000.0
# 沿现状中线采样 DEM 地面高程(相对 0 m 基准)
gz = dem.ground_elev_xy(a['X'], a['Y'], math.radians(lat0d), lon0d) \
    if False else dem.ground_elev_xy(a['X'], a['Y'], lat0d, lon0d)

fig, ax = plt.subplots(figsize=(14, 5.2), dpi=140)
ax.fill_between(s_km, 0, gz, where=gz >= 0, color='#9fdf9f', alpha=.6, label='above baseline')
ax.fill_between(s_km, 0, gz, where=gz < 0, color='#92c5de', alpha=.6, label='below baseline')
ax.plot(s_km, gz, color='#8a5a2b', lw=2.0, label='ground elevation along alignment')
ax.axhline(0, color='k', lw=2.0, label='0 m baseline')
# 标注最高/最低点
i_hi = int(np.argmax(gz)); i_lo = int(np.argmin(gz))
ax.annotate('max %.0f m (K%.1f)' % (gz[i_hi], s_km[i_hi]),
            (s_km[i_hi], gz[i_hi]), textcoords='offset points', xytext=(0, 8), fontsize=9)
ax.annotate('min %.0f m (K%.1f)' % (gz[i_lo], s_km[i_lo]),
            (s_km[i_lo], gz[i_lo]), textcoords='offset points', xytext=(0, -14), fontsize=9)
ax.set_xlabel('mileage along plan alignment (km)')
ax.set_ylabel('ground elevation vs 0 m baseline (m)')
ax.set_title('Fig 2b  Ground elevation trend along the plan alignment (0 m baseline)')
ax.grid(alpha=.3); ax.legend(loc='upper right', fontsize=9)
fig.tight_layout(); fig.savefig('../figures/region_fig2b_contours.png', facecolor='white')
print('saved; ground elev range %.1f ~ %.1f m' % (gz.min(), gz.max()))
