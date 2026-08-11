# -*- coding: utf-8 -*-
"""draw_contour2d.py — 建设区域二维等高线图(X,Y 平面), 20 m 等高距 + 线位叠加。"""
import numpy as np, logging, math
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
from scipy.ndimage import gaussian_filter
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from data_loader import load_alignment

R_E = 6378137.0
a = load_alignment(); X, Y = a['X'], a['Y']
lat0 = math.radians(a['lat'][0]); lon0 = math.radians(a['lon'][0])

d = np.load('dem_xwide_z14.npz')
E = d['elev']; Z = int(d['z']); x0 = int(d['x0']); y0 = int(d['y0'])
H, Wd = E.shape; n = 2**Z
lon_px = (x0+(np.arange(Wd)+0.5)/256.0)/n*360.0-180.0
lat_rd = np.arctan(np.sinh(np.pi*(1-2*(y0+(np.arange(H)+0.5)/256.0)/n)))
GLON, GLAT = np.meshgrid(np.radians(lon_px), lat_rd)
DX = R_E*math.cos(lat0)*(GLON-lon0); DY = R_E*(GLAT-lat0)
Em = np.where(E < -100, np.nan, E)
Esm = gaussian_filter(np.nan_to_num(Em, nan=float(np.nanmedian(Em))), sigma=2.5)

fig, ax = plt.subplots(figsize=(14, 7), dpi=140)
ax.pcolormesh(DX, DY, Em, cmap='terrain', shading='auto', alpha=.45, zorder=0)
levels = np.arange(0, 361, 20)               # 20 m 等高距
cf = ax.contourf(DX, DY, Esm, levels=levels, cmap='terrain', alpha=.55, zorder=1)
cb = fig.colorbar(cf, ax=ax, shrink=.82); cb.set_label('ground elevation (m)')
cs = ax.contour(DX, DY, Esm, levels=levels, colors='k', linewidths=0.5, alpha=.6, zorder=2)
ax.clabel(cs, levels[::2], fmt='%d', fontsize=6, inline=True)
c0 = ax.contour(DX, DY, Esm, levels=[0], colors='#b2182b', linewidths=1.6, zorder=3)

ax.plot(X, Y, color='#ffe400', lw=2.4, zorder=4, label='centerline')
ax.plot(X[0], Y[0], 'o', ms=10, color='green', zorder=5)
ax.plot(X[-1], Y[-1], 's', ms=9, color='red', zorder=5)
ax.set_aspect('equal'); ax.set_xlim(-20800, 800); ax.set_ylim(-1400, 3000)
ax.set_xlabel('X East (m)'); ax.set_ylabel('Y North (m)')
ax.set_title('Fig 2c  2D contour map of the construction region (20 m interval, 0 m in red)')
ax.legend(loc='upper right', fontsize=9)
fig.tight_layout(); fig.savefig('../figures/region_fig2c_contour2d.png', facecolor='white')
print('saved region_fig2c_contour2d.png')
