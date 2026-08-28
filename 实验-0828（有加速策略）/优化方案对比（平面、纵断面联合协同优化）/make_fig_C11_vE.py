# -*- coding: utf-8 -*-
"""版本E: 白云山近景(东段约10km窗口) + 低视角 — 突出隧道规避与山体关系。"""
from fig3d_core import render
import fig3d_core as F

# 取包络框中部偏东 10 km 窗口(白云山主峰所在), 世界坐标由数据范围推算
x_mid0 = F.X0 + (F.X1 - F.X0) * 0.30
x_mid1 = F.X0 + (F.X1 - F.X0) * 0.72
render(dict(
    tag="vE", name="白云山近景",
    terrain=["#f4f1ea", "#e4ddc7", "#cfc09a", "#b09b6d", "#8d7550", "#e8e4dd"],
    line_colors=("#b6321f", "#e88f1a", "#5b3a8e"),
    elev=30, azim=-98, extent=(x_mid0, x_mid1), z_vis=3.0,
    figsize=(11.5, 5.6), title_y=0.92, legend_at=(0.02, 0.86),
))
