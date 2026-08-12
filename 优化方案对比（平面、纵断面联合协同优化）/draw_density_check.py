# -*- coding: utf-8 -*-
"""
draw_density_check.py — 密度分区约束的核验图: 最终线位 + Tier1/Tier2 分区叠加

用途: 目视确认 (a) 分区栅格与线位落在同一坐标系, (b) 线位未穿越 Tier2 禁行区。
配合 run_joint.py 输出的 L_dense2_km 断言使用(数值为 0 且图上不相交才算通过)。
"""
import json
import os
import sys

import numpy as np
import logging
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import building_mask as bm

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results',
                   sys.argv[1] if len(sys.argv) > 1 else 'joint_results.json')
FIG = os.path.join(HERE, 'figures')
os.makedirs(FIG, exist_ok=True)

t = bm._load()
tier1 = t['tier1']
tier2 = t['tier2']
X0, Y0, CELL = t['X0'], t['Y0'], t['CELL']
NX, NY = t['NX'], t['NY']
ext = [X0, X0 + NX * CELL, Y0, Y0 + NY * CELL]

d = json.load(open(RES, encoding='utf-8'))
A, C = d['M_A'], d['M_C']
ax_, ay_ = np.array(A['plane_x']), np.array(A['plane_y'])
cx_, cy_ = np.array(C['plane_x']), np.array(C['plane_y'])

fig, ax = plt.subplots(figsize=(15.2, 5.6), dpi=140)
ax.set_rasterization_zorder(3)
ax.imshow(t['S'], origin='lower', extent=ext, cmap='Greys', zorder=0,
          vmin=0, vmax=float(np.percentile(t['S'][t['S'] > 0], 99)))
ov = np.zeros(tier1.shape, dtype=int)
ov[tier1] = 1
ov[tier2] = 2
ax.imshow(np.ma.masked_where(ov == 0, ov), origin='lower', extent=ext,
          cmap=ListedColormap(['#f4a582', '#b2182b']), vmin=1, vmax=2,
          alpha=0.72, zorder=1, interpolation='nearest')
ax.plot(ax_, ay_, color='#222222', lw=1.4, zorder=4)
ax.plot(cx_, cy_, color='#0050ff', lw=2.4, zorder=5)

ax.set_aspect('equal')
ax.set_xlim(cx_.min() - 2500, cx_.max() + 2500)
ax.set_ylim(cy_.min() - 3000, cy_.max() + 3000)
ax.set_xlabel('X East (m)')
ax.set_ylabel('Y North (m)')
tp, tf, dmax = bm.thresholds()
ax.set_title('Density-zoning feasibility check  |  '
             f'theta_forbid={tf:.4f} (= D_A_max {dmax:.4f} x 1.15)  |  '
             f"M-C: Tier2 {C['L_dense2_km']:.3f} km (must be 0), "
             f"Tier1 {C['L_dense1_km']:.2f} km")
ax.legend(handles=[
    Patch(facecolor='#b2182b', label='Tier2 forbidden (hard constraint)'),
    Patch(facecolor='#f4a582', label='Tier1 passable with soft penalty'),
    Line2D([0], [0], color='#0050ff', lw=2.4,
           label=f"M-C optimized ({C['L_km']:.2f} km)"),
    Line2D([0], [0], color='#222222', lw=1.4,
           label=f"M-A existing ({A['L_km']:.2f} km)"),
], loc='lower left', ncol=2, fontsize=9, framealpha=0.9)
fig.tight_layout()
out = os.path.join(FIG, '图C10_密度分区可行性核验.png')
fig.savefig(out, facecolor='white')
plt.close(fig)
print(f'[图] {os.path.basename(out)}')
print(f"  M-A: Tier2 {A['L_dense2_km']:.3f} km  Tier1 {A['L_dense1_km']:.2f} km")
print(f"  M-C: Tier2 {C['L_dense2_km']:.3f} km  Tier1 {C['L_dense1_km']:.2f} km")
