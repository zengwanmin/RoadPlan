# -*- coding: utf-8 -*-
"""版本A: 学术米色晕渲(基准版) — 米白-卡其地形, 标准斜视角。"""
from fig3d_core import render

render(dict(
    tag="vA", name="学术米色",
    terrain=["#f4f1ea", "#e4ddc7", "#cfc09a", "#b09b6d", "#8d7550", "#e8e4dd"],
    line_colors=("#b6321f", "#e88f1a", "#5b3a8e"),
    elev=42, azim=-112,
))
