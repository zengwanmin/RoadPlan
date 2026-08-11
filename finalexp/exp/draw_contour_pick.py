# -*- coding: utf-8 -*-
"""draw_contour_pick.py — 只标注特定等高线(0 与 ±20 / 0 与 ±30 m)两张图。"""
import numpy as np, logging, math
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
from scipy.ndimage import gaussian_filter
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

R_E = 6378137.0
from data_loader import load_alignment
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


def plot_pick(lv, out, tag):
    fig, ax = plt.subplots(figsize=(14, 7), dpi=140)
    ax.pcolormesh(DX, DY, Em, cmap='terrain', shading='auto', alpha=.55, zorder=0)
    cb = fig.colorbar(ax.collections[0], ax=ax, shrink=.82); cb.set_label('ground elevation (m)')
    c0 = ax.contour(DX, DY, Esm, levels=[0], colors='k', linewidths=2.0, zorder=3)
    ax.clabel(c0, fmt='0 m', fontsize=8, inline=True)
    cp = ax.contour(DX, DY, Esm, levels=[-lv, lv], colors=['#2166ac', '#b2182b'],
                    linewidths=1.4, zorder=2)
    ax.clabel(cp, fmt='%d', fontsize=8, inline=True)
    ax.plot(X, Y, color='#ffe400', lw=2.4, zorder=4, label='centerline')
    ax.plot(X[0], Y[0], 'o', ms=10, color='green', zorder=5)
    ax.plot(X[-1], Y[-1], 's', ms=9, color='red', zorder=5)
    ax.set_aspect('equal'); ax.set_xlim(-20800, 800); ax.set_ylim(-1400, 3000)
    ax.set_xlabel('X East (m)'); ax.set_ylabel('Y North (m)')
    ax.set_title('%s  2D contour: 0 m (black) and +/-%d m (red/blue)' % (tag, lv))
    ax.legend(loc='upper right', fontsize=9)
    fig.tight_layout(); fig.savefig('../figures/'+out, facecolor='white'); plt.close(fig)
    print('saved', out)


plot_pick(20, 'region_fig2d_contour_pm20.png', 'Fig 2d')
plot_pick(30, 'region_fig2e_contour_pm30.png', 'Fig 2e')
