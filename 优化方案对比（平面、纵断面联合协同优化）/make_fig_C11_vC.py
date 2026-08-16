# -*- coding: utf-8 -*-
"""版本C: 经典地形绿棕 + 高视角(近俯视) — 地图感强, 平面走向最清楚。"""
from fig3d_core import render

render(dict(
    tag="vC", name="经典地形高视角",
    terrain=["#e9f0e3", "#cfe0bd", "#a9c98e", "#c9b986", "#a08252", "#f0ede6"],
    line_colors=("#c62828", "#ef8f00", "#6a1b9a"),
    elev=62, azim=-105, z_vis=2.6, title_y=0.92,
    legend_at=(0.02, 0.86),
))
