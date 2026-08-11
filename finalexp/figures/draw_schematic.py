# -*- coding: utf-8 -*-
"""draw_schematic.py — 问题定义与抽象原理图(卡通手绘风, xkcd 模式)。"""
import numpy as np
import logging
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle, Wedge, Rectangle, Polygon

plt.xkcd(scale=1.2, length=120, randomness=1.6)
fig = plt.figure(figsize=(15.5, 10.5), dpi=140)
gs = fig.add_gridspec(2, 3, width_ratios=[2.1, 2.1, 1.25], height_ratios=[1, 1],
                      hspace=0.42, wspace=0.25)
axP = fig.add_subplot(gs[0, :2])
axV = fig.add_subplot(gs[1, :2])
axL = fig.add_subplot(gs[:, 2])
axL.axis('off')

# ---------------- PLAN VIEW ----------------
t = np.linspace(0, 1, 400)
ex_y = 0.28*np.sin(3.1*np.pi*t) + 0.11*np.sin(7*np.pi*t) + 0.05*np.sin(13*np.pi*t)
op_y = 0.55*ex_y
axP.fill_between(t, ex_y-0.28, ex_y+0.28, color='#ffe9a8', alpha=.8, zorder=0)
axP.plot(t, ex_y, 'k--', lw=2, label='existing road')
axP.plot(t, op_y, color='#1f6fd6', lw=3.2, label='optimized alignment')
axP.annotate('corridor +/- W', (0.75, 0.52), fontsize=13, color='#b8860b')
axP.plot(0, 0, marker='>', ms=16, color='green')
axP.annotate('A start', (-0.02, -0.16), fontsize=12, color='green')
axP.plot(1, ex_y[-1], marker='s', ms=12, color='red')
axP.annotate('B end', (0.93, ex_y[-1]-0.18), fontsize=12, color='red')
cx, cy = 0.16, 0.68
axP.add_patch(Circle((cx, cy), 0.13, fill=False, lw=2, color='#d64541'))
th = np.linspace(0.6, 2.5, 60)
axP.plot(cx+0.075*np.cos(th), cy-0.03+0.075*np.sin(th), color='#1f6fd6', lw=2.5)
axP.add_patch(Circle((cx, cy-0.03), 0.075, fill=False, ls=':', lw=1.6, color='#d64541'))
axP.annotate('R >= 400 m\n(no sharp turns!)', (cx+0.15, cy+0.03), fontsize=12, color='#d64541')
axP.plot([0.30, cx+0.10], [op_y[120]+0.01, cy-0.10], color='#d64541', lw=1.2)
mx = 0.70
axP.add_patch(Polygon([[mx-0.07, -0.05], [mx, 0.20], [mx+0.07, -0.05]], closed=True,
                      fc='#c9a36a', ec='k', lw=1.5, zorder=1))
axP.add_patch(Wedge((mx, -0.02), 0.03, 0, 180, fc='k', zorder=2))
axP.annotate('tunnel', (mx+0.06, 0.10), fontsize=11)
bxc = 0.44
bidx = int(bxc*399)
axP.plot([bxc-0.03, bxc+0.03], [op_y[bidx]]*2, lw=5, color='#8a5a2b', solid_capstyle='butt')
for bb in (-0.02, 0, 0.02):
    axP.plot([bxc+bb]*2, [op_y[bidx]-0.05, op_y[bidx]], lw=2, color='#8a5a2b')
axP.annotate('bridge', (bxc-0.04, op_y[bidx]-0.15), fontsize=11, color='#8a5a2b')
axP.set_title('PLAN VIEW  (x, y Cartesian)', fontsize=16)
axP.set_xlabel('x')
axP.set_ylabel('y')
axP.set_xlim(-0.06, 1.28)
axP.set_ylim(-0.45, 0.95)
axP.legend(loc='lower right', fontsize=10)

# ---------------- PROFILE VIEW ----------------
M = np.linspace(0, 1, 400)
ground = 0.25 + 0.16*np.sin(2.4*np.pi*M) + 0.07*np.sin(6.3*np.pi*M) \
    + 0.28*np.exp(-((M-0.68)/0.07)**2)
pvi_M = np.array([0, 0.18, 0.35, 0.52, 0.63, 0.74, 0.88, 1.0])
pvi_H = np.array([0.28, 0.33, 0.22, 0.30, 0.42, 0.44, 0.30, 0.27])
design = np.interp(M, pvi_M, pvi_H)
axV.plot(M, ground, color='#8a5a2b', lw=2.4, label='terrain g(M)')
axV.plot(pvi_M, pvi_H, color='#1f6fd6', lw=3, marker='o', ms=7, label='design profile (PVI)')
axV.fill_between(M, ground, design, where=design < ground, color='#ff9d9d',
                 alpha=.55, hatch='///', label='cut')
axV.fill_between(M, ground, design, where=design >= ground, color='#9fdf9f',
                 alpha=.55, hatch='\\\\\\', label='fill')
px, ph = pvi_M[2], pvi_H[2]
axV.add_patch(Wedge((px, ph), 0.10, -14, 14, fc='#ffd27f', alpha=.9))
axV.annotate('slope <= 4%', (px+0.02, ph-0.10), fontsize=12, color='#b8860b')
kx, kh = pvi_M[5], pvi_H[5]
axV.add_patch(Circle((kx, kh-0.035), 0.045, fill=False, ls=':', lw=2, color='#7d3cc9'))
axV.annotate('vertical curve  L >= kA', (kx+0.05, kh+0.05), fontsize=11, color='#7d3cc9')
axV.add_patch(Rectangle((0.44, 0.02), 0.12, 0.035, fc='none', ec='#e67e22',
                        hatch='xxx', lw=1.6))
axV.annotate('no-PVI zone (circular curve in plan)', (0.30, -0.11), fontsize=11,
             color='#e67e22')
axV.set_title('PROFILE VIEW  (M = mileage along plan,  H = elevation)', fontsize=15)
axV.set_xlabel('M  (defined BY the plan curve!)')
axV.set_ylabel('H')
axV.set_xlim(-0.05, 1.28)
axV.set_ylim(-0.16, 0.72)
axV.legend(loc='upper left', fontsize=9, ncol=2)

# ---------------- coupling arrows ----------------
fig.patches.append(FancyArrowPatch((0.315, 0.545), (0.315, 0.475),
                   transform=fig.transFigure, arrowstyle='-|>', mutation_scale=34,
                   lw=3.5, color='#e67e22'))
fig.text(0.325, 0.505, 'plan defines mileage axis M\n& ground line g(M)',
         fontsize=12, color='#e67e22')
fig.patches.append(FancyArrowPatch((0.205, 0.545), (0.205, 0.475),
                   transform=fig.transFigure, arrowstyle='-|>', mutation_scale=34,
                   lw=3.5, color='#e67e22', linestyle=(0, (6, 3))))

# ---------------- SOLVER LOOP ----------------
axL.set_xlim(0, 1)
axL.set_ylim(-0.06, 1)
axL.set_title('SOLVER LOOP', fontsize=15)
jx, jy = 0.42, 0.82
axL.add_patch(Wedge((jx, jy), 0.11, 0, 180, fc='#c9a0f0', ec='k', lw=1.6))
for dx in (-0.07, -0.025, 0.025, 0.07):
    s = np.linspace(0, 1, 40)
    axL.plot(jx+dx+0.013*np.sin(9*s), jy-0.005-0.12*s, color='#9b59b6', lw=2)
axL.plot([jx-0.04, jx+0.04], [jy+0.045]*2, 'k.', ms=4)
axL.text(0.42, 0.965, 'IJS searches the PLAN\n(40 sine modes)', ha='center', fontsize=12)
axL.add_patch(Rectangle((0.18, 0.44), 0.48, 0.13, fc='#bfe3ff', ec='k', lw=1.6))
axL.text(0.42, 0.505, 'DP: exact PROFILE\nfor each plan', ha='center', va='center',
         fontsize=12)
bx, by = 0.42, 0.13
axL.plot([bx, bx], [by, by+0.12], color='k', lw=2.5)
axL.plot([bx-0.16, bx+0.16], [by+0.12]*2, color='k', lw=2.5)
for sx, lab, col in [(-0.16, 'C', '#ffe08a'), (0.16, 'E', '#a8e6a1')]:
    axL.plot([bx+sx]*2, [by+0.12, by+0.065], color='k', lw=1.5)
    axL.add_patch(Circle((bx+sx, by+0.02), 0.048, fc=col, ec='k'))
    axL.text(bx+sx, by+0.02, lab, ha='center', va='center', fontsize=13)
axL.text(0.42, -0.045, 'entropy weights  wC : wE', ha='center', fontsize=12)
for (x1, y1, x2, y2) in [(0.42, 0.68, 0.42, 0.585), (0.42, 0.42, 0.42, 0.30)]:
    axL.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>',
                  mutation_scale=28, lw=3, color='#444'))
axL.add_patch(FancyArrowPatch((0.72, 0.16), (0.72, 0.80), arrowstyle='-|>',
              mutation_scale=28, lw=3, color='#444',
              connectionstyle='arc3,rad=-0.4'))
axL.text(0.95, 0.48, 'F = wC*C/Cref + wE*E/Eref', rotation=90, va='center', fontsize=11)

fig.suptitle('Highway 3D Alignment Optimization -- problem abstraction\n'
             '(two coupled Cartesian frames + bilevel solver)', fontsize=17, y=1.00)
fig.savefig('problem_schematic_cartoon.png', bbox_inches='tight', facecolor='white')
print('saved v2')
