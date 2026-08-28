# -*- coding: utf-8 -*-
"""版本D: 深色主题 — 深蓝灰地形, 亮色线形, 演示/封面风格。"""
from fig3d_core import render

render(dict(
    tag="vD", name="深色主题",
    terrain=["#1c2733", "#26394a", "#33506a", "#4a6a7d", "#7c8f8a", "#404a52"],
    line_colors=("#ff5252", "#ffb300", "#ce93f8"),
    ma_color="#9aa4ad", dark=True,
    elev=42, azim=-112, light_alt=38,
))
