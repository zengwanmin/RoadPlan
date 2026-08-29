# -*- coding: utf-8 -*-
"""
draw_density_check.py — 图C10建筑密度分布叠加: 最终线位 + Tier1/Tier2分区

建筑密度不进入最终主实验的目标或约束。本图只把建筑分布作为地理背景叠加，
用于增强方案可视化和观察线位与建成区的空间关系，不作可行性判定。
"""
import json
import hashlib
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
                   sys.argv[1] if len(sys.argv) > 1 else 'joint_results_w500_nodens.json')
FIG = os.path.join(HERE, 'figures')
os.makedirs(FIG, exist_ok=True)

t = bm._load()
tier1 = t['tier1']
tier2 = t['tier2']
X0, Y0, CELL = t['X0'], t['Y0'], t['CELL']
NX, NY = t['NX'], t['NY']
ext = [X0, X0 + NX * CELL, Y0, Y0 + NY * CELL]

d = json.load(open(RES, encoding='utf-8'))
meta = d.get('meta', {})
if (meta.get('density_on') is not False or
        meta.get('profile_endpoints_fixed') is not True):
    raise RuntimeError('Fig. C10 requires fixed-endpoint, density-disabled main results')
if 'M_C' not in d:
    # 公平Pareto口径下，联合原始结果只保存前沿；最终联合 M-C
    # 保存在与该前沿绑定的两阶段结果公共决策中。
    two_name = os.path.basename(RES).replace('joint_results', 'twostage_results', 1)
    two_path = os.path.join(HERE, 'results', two_name)
    with open(two_path, encoding='utf-8') as fp:
        two = json.load(fp)
    h = hashlib.sha256()
    with open(RES, 'rb') as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b''):
            h.update(chunk)
    fair = two.get('fair_decision', {})
    if fair.get('joint_result_sha256') != h.hexdigest():
        raise RuntimeError('Fig. C10 two-stage decision is not bound to this joint front')
    d['M_C'] = fair['joint']['M_C']
A, C = d['M_A'], d['M_C']
ax_, ay_ = np.array(A['plane_x']), np.array(A['plane_y'])
cx_, cy_ = np.array(C['plane_x']), np.array(C['plane_y'])


def overlap_lengths_km(scheme):
    """独立从建筑分区图层计算叠加长度；该诊断量不参与优化。"""
    x = np.asarray(scheme['plane_x'], float)
    y = np.asarray(scheme['plane_y'], float)
    sta = np.asarray(scheme['sta'], float)
    if not (len(x) == len(y) == len(sta)) or len(sta) < 2:
        raise ValueError('Invalid alignment arrays in main result')

    def integrate(mask):
        values = np.asarray(mask, dtype=float)
        return float(np.sum(0.5 * (values[:-1] + values[1:]) * np.diff(sta))) / 1000.0

    return dict(tier1=integrate(bm.tier1_at_xy(x, y)),
                tier2=integrate(bm.tier2_at_xy(x, y)))


overlap_A = overlap_lengths_km(A)
overlap_C = overlap_lengths_km(C)

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
ax.set_title('Building-density overlay (visualization only; not an optimization constraint)  |  '
             f'theta_forbid={tf:.4f} (= D_A_max {dmax:.4f} x 1.15)  |  '
             f"M-C overlap: Tier2 {overlap_C['tier2']:.3f} km, "
             f"Tier1 {overlap_C['tier1']:.2f} km")
ax.legend(handles=[
    Patch(facecolor='#b2182b', label='Tier2 high-density area (visual context)'),
    Patch(facecolor='#f4a582', label='Tier1 medium-density area (visual context)'),
    Line2D([0], [0], color='#0050ff', lw=2.4,
           label=f"M-C optimized ({C['L_km']:.2f} km)"),
    Line2D([0], [0], color='#222222', lw=1.4,
           label=f"M-A existing ({A['L_km']:.2f} km)"),
], loc='lower left', ncol=2, fontsize=9, framealpha=0.9)
fig.tight_layout()
out = os.path.join(FIG, '图C10_建筑密度分布叠加_仅可视化.png')
fig.savefig(out, facecolor='white')
plt.close(fig)
print(f'[图] {os.path.basename(out)}')
print(f"  M-A: Tier2 {overlap_A['tier2']:.3f} km  Tier1 {overlap_A['tier1']:.2f} km")
print(f"  M-C: Tier2 {overlap_C['tier2']:.3f} km  Tier1 {overlap_C['tier1']:.2f} km")
